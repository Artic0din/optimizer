"""Sign in with ChatGPT: plan explanations on the operator's ChatGPT plan instead of an Azure deployment.

Follows OpenAI's public-client flow for ChatGPT plan usage: dynamic client registration with PKCE, a
loopback redirect and refresh token rotation. A headless host never receives the 127.0.0.1 callback, so
the operator pastes the URL the browser lands on back in.

Every gunicorn worker reads the same credentials file. Changes happen under an exclusive lock and are
written atomically, so the pending sign-in and a rotating refresh token are shared and never raced.
"""
import base64
import fcntl
import hashlib
import hmac
import http.client
import json
import os
import secrets
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from contextlib import contextmanager
from urllib.parse import parse_qs, urlencode, urlsplit

import jwt

ISSUER = 'https://auth.openai.com'
AUTHORIZE_URL = ISSUER + '/api/accounts/authorize'
TOKEN_URL = ISSUER + '/api/accounts/oauth/token'
JWKS_URL = ISSUER + '/.well-known/jwks.json'
REVOKE_URL = ISSUER + '/api/accounts/oauth/revoke'
RESOURCE = 'https://api.openai.com/v1'
RESPONSES_URL = RESOURCE + '/responses'
REDIRECT_URI = 'http://127.0.0.1:1455/auth/callback'
SCOPES = 'openid profile email offline_access resource.invoke chatgpt.tokens.use.direct'
PLAN_SCOPE = 'chatgpt.tokens.use.direct'
DYNAMIC_CLIENT = 'dynamic_agent_client'
AGENT_NAME = 'evcc optimizer'

LOGIN_SECONDS = 600
REFRESH_MARGIN = 180  # seconds before expiry an access token is renewed
HTTP_TIMEOUT = 20
RESPONSES_TIMEOUT = 45  # same as the Azure call
TOKEN_FIELDS = ('access_token', 'refresh_token', 'id_token', 'expires_at', 'scopes')
# answers that mean the grant is gone for good, anything else is worth a retry later
TERMINAL_REFRESH_ERRORS = {'invalid_grant', 'invalid_client', 'unauthorized_client', 'access_denied', 'invalid_refresh_token',
                           'token_expired', 'refresh_token_expired', 'refresh_token_invalidated', 'refresh_token_reused'}
REVOCATION_UNCONFIRMED = 'Signed out here, but ChatGPT did not confirm the session was revoked. Disconnect the app in ChatGPT settings'


class ChatGPTError(Exception):
    """A failure safe to show the operator: never carries a token, code or callback URL."""

    def __init__(self, message: str, status: int = 502):
        super().__init__(message)
        self.status = status


def configured(settings) -> bool:
    return bool(settings.chatgpt_credentials)


def connected(settings) -> bool:
    return configured(settings) and bool(_read(settings).get('refresh_token'))


@contextmanager
def _locked(settings):
    """The record under an exclusive lock; the caller writes back what it changes."""
    with open(settings.chatgpt_credentials + '.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield _read(settings)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _read(settings) -> dict:
    try:
        with open(settings.chatgpt_credentials) as f:
            record = json.load(f)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        raise ChatGPTError('ChatGPT credentials could not be read', 503) from e
    if not isinstance(record, dict):
        raise ChatGPTError('ChatGPT credentials could not be read', 503)
    return record


def _write(settings, record: dict) -> None:
    folder = os.path.dirname(os.path.abspath(settings.chatgpt_credentials))
    fd, tmp = tempfile.mkstemp(dir=folder, prefix='.chatgpt-')
    try:
        with os.fdopen(fd, 'w') as f:
            os.fchmod(f.fileno(), 0o600)
            json.dump(record, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, settings.chatgpt_credentials)
    except OSError as e:
        os.unlink(tmp)
        raise ChatGPTError('ChatGPT credentials could not be saved', 503) from e


def _clear_tokens(record: dict) -> dict:
    """The registration without its tokens: client and host stay for the next sign-in."""
    return {k: v for k, v in record.items() if k not in TOKEN_FIELDS}


def _post_form(url: str, data: dict) -> dict:
    request = urllib.request.Request(url, data=urlencode(data).encode(), method='POST',
                                     headers={'Content-Type': 'application/x-www-form-urlencoded', 'Accept': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        try:
            error = json.loads(e.read()).get('error')
        except (ValueError, AttributeError):
            error = None
        if isinstance(error, dict):
            error = error.get('code') or error.get('type')
        raise ChatGPTError(f'ChatGPT returned {e.code}' + (f': {error}' if isinstance(error, str) else ''),
                           401 if error in TERMINAL_REFRESH_ERRORS else 502) from None
    except (urllib.error.URLError, TimeoutError) as e:
        raise ChatGPTError(f'ChatGPT unreachable: {getattr(e, "reason", e)}') from None
    try:
        return json.loads(raw) if raw else {}
    except ValueError:
        raise ChatGPTError('ChatGPT returned an invalid response') from None


def _token_fields(data: dict, scopes: list | None = None) -> dict:
    granted = data['scope'].split() if isinstance(data.get('scope'), str) else scopes or []
    if PLAN_SCOPE not in granted:
        raise ChatGPTError('ChatGPT plan usage was not granted. Sign in again and allow it', 403)
    if not data.get('access_token') or not data.get('refresh_token') or not isinstance(data.get('expires_in'), (int, float)):
        raise ChatGPTError('ChatGPT returned incomplete credentials')
    fields = {'access_token': data['access_token'], 'refresh_token': data['refresh_token'],
              'expires_at': time.time() + data['expires_in'], 'scopes': granted}
    if data.get('id_token'):
        fields['id_token'] = data['id_token']
    return fields


def _identity(id_token: str, client_id: str, nonce: str | None) -> dict:
    """The verified claims of an ID token: OpenAI's signature, issuer, audience, expiry and, at sign-in, the nonce."""
    try:
        key = jwt.PyJWKClient(JWKS_URL, timeout=HTTP_TIMEOUT).get_signing_key_from_jwt(id_token).key
        claims = jwt.decode(id_token, key, algorithms=['RS256'], audience=client_id, issuer=ISSUER)
    except jwt.PyJWTError as e:
        raise ChatGPTError(f'ChatGPT ID token rejected: {type(e).__name__}') from None
    if nonce is not None and not hmac.compare_digest(str(claims.get('nonce', '')), nonce):
        raise ChatGPTError('ChatGPT ID token rejected: nonce mismatch')
    if not claims.get('sub'):
        raise ChatGPTError('ChatGPT ID token rejected: no subject')
    return claims


def today() -> str:
    """The local date the daily limit counts on, in the container's TZ."""
    return time.strftime('%Y-%m-%d')


def _spend(settings) -> None:
    """Count one explanation against the daily limit, before the request: a failed one still cost a request."""
    limit = settings.chatgpt_daily_limit
    if not limit:
        return
    with _locked(settings) as record:
        usage = record.get('usage') if isinstance(record.get('usage'), dict) else {}
        count = usage.get('count', 0) if usage.get('date') == today() else 0
        if count >= limit:
            raise ChatGPTError(f'The daily limit of {limit} ChatGPT explanations is reached', 429)
        _write(settings, {**record, 'usage': {'date': today(), 'count': count + 1}})


def status(settings) -> dict:
    record = _read(settings)
    if not record.get('refresh_token'):
        return {'connected': False}
    return {'connected': True, 'email': record.get('email'), 'expires_at': record.get('expires_at')}


def start_login(settings) -> dict:
    """The authorization URL for the operator's browser. A new login replaces any pending one."""
    with _locked(settings) as record:
        record.setdefault('ext_agent_host_id', f'urn:uuid:{uuid.uuid4()}')
        verifier = secrets.token_urlsafe(48)
        pending = {'state': secrets.token_urlsafe(32), 'nonce': secrets.token_urlsafe(32), 'verifier': verifier,
                   'expires_at': time.time() + LOGIN_SECONDS}
        params = {
            'client_id': record.get('client_id') or DYNAMIC_CLIENT,
            'response_type': 'code',
            'redirect_uri': REDIRECT_URI,
            'scope': SCOPES,
            'resource': RESOURCE,
            'state': pending['state'],
            'nonce': pending['nonce'],
            'code_challenge_method': 'S256',
            'code_challenge': base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b'=').decode(),
            'ext_agent_host_id': record['ext_agent_host_id'],
        }
        if 'client_id' in record:
            if record.get('id_token'):
                params['id_token_hint'] = record['id_token']
            if record.get('email'):
                params['login_hint'] = record['email']
        else:
            params['agent_name_hint'] = AGENT_NAME
        _write(settings, {**record, 'pending': pending})
    return {'auth_url': AUTHORIZE_URL + '?' + urlencode(params), 'expires_at': pending['expires_at']}


def complete_login(settings, callback_url: str) -> dict:
    """Exchange the code in the pasted callback URL and keep the verified credentials."""
    with _locked(settings) as record:
        pending = record.get('pending')
        if not pending or pending['expires_at'] < time.time():
            raise ChatGPTError('The ChatGPT sign-in expired. Start it again', 400)
        parsed = urlsplit(callback_url.strip())
        params = {k: v[0] for k, v in parse_qs(parsed.query).items()}
        if (f'{parsed.scheme}://{parsed.netloc}{parsed.path}' != REDIRECT_URI
                or not hmac.compare_digest(params.get('state', '').encode(), pending['state'].encode())):
            raise ChatGPTError('The callback URL does not belong to the pending sign-in', 400)
        if 'error' in params:
            _write(settings, {k: v for k, v in record.items() if k != 'pending'})
            raise ChatGPTError('The ChatGPT sign-in was declined', 400)
        client_id = params.get('client_id') or record.get('client_id')
        if not params.get('code') or not client_id or client_id == DYNAMIC_CLIENT:
            raise ChatGPTError('ChatGPT registration was incomplete. Start the sign-in again', 400)
        if record.get('client_id') and client_id != record['client_id']:
            raise ChatGPTError('The callback belongs to another ChatGPT registration', 400)

        # keep the issued client even if the exchange fails: the next attempt reuses it
        registered = {**record, 'client_id': client_id}
        _write(settings, registered)
        try:
            data = _post_form(TOKEN_URL, {'grant_type': 'authorization_code', 'client_id': client_id, 'code': params['code'],
                                          'code_verifier': pending['verifier'], 'redirect_uri': REDIRECT_URI, 'resource': RESOURCE})
        except ChatGPTError as e:
            if e.status == 401:
                raise ChatGPTError('ChatGPT rejected the sign-in code. Start the sign-in again', 400) from None
            raise
        fields = _token_fields(data)
        claims = _identity(data.get('id_token') or '', client_id, pending['nonce'])
        if registered.get('subject') and claims['sub'] != registered['subject']:
            raise ChatGPTError('Sign in with the ChatGPT account already connected', 400)
        kept = {k: v for k, v in _clear_tokens(registered).items() if k != 'pending'}
        _write(settings, {**kept, **fields, 'subject': claims['sub'], 'email': claims.get('email')})
    return status(settings)


def access_token(settings, force: bool = False) -> str:
    """A valid access token, renewed shortly before expiry or when the last one was rejected."""
    with _locked(settings) as record:
        if not record.get('refresh_token'):
            raise ChatGPTError('Sign in with ChatGPT first', 503)
        if not force and record['expires_at'] > time.time() + REFRESH_MARGIN:
            return record['access_token']
        try:
            data = _post_form(TOKEN_URL, {'grant_type': 'refresh_token', 'client_id': record['client_id'],
                                          'refresh_token': record['refresh_token'], 'resource': RESOURCE})
            fields = _token_fields(data, record.get('scopes'))
        except ChatGPTError as e:
            if e.status in (401, 403):
                _write(settings, _clear_tokens(record))
                raise ChatGPTError('The ChatGPT connection is no longer valid. Sign in again', 503) from None
            raise
        # the old refresh token is spent: persist the replacement before anything can fail
        _write(settings, {**record, **fields})
        return fields['access_token']


def logout(settings) -> dict:
    """Revoke the renewable session and forget the tokens, keeping the registration for the next sign-in."""
    result = {'connected': False}
    with _locked(settings) as record:
        if record.get('refresh_token'):
            try:
                _post_form(REVOKE_URL, {'token': record['refresh_token'], 'token_type_hint': 'refresh_token', 'client_id': record['client_id']})
            except ChatGPTError:
                result['warning'] = REVOCATION_UNCONFIRMED
        _write(settings, {k: v for k, v in _clear_tokens(record).items() if k != 'pending'})
    return result


def _stream(token: str, body: dict):
    request = urllib.request.Request(RESPONSES_URL, data=json.dumps(body).encode(), method='POST', headers={
        'Content-Type': 'application/json', 'Accept': 'text/event-stream', 'Authorization': f'Bearer {token}'})
    return urllib.request.urlopen(request, timeout=RESPONSES_TIMEOUT)


def respond(settings, instructions: str, text: str) -> tuple[str, dict]:
    """One Responses API call on the ChatGPT plan. Returns the text and the usage once the response completed."""
    body = {'model': settings.chatgpt_model, 'instructions': instructions, 'input': [{'role': 'user', 'content': text}],
            'reasoning': {'effort': 'low'}, 'store': False, 'stream': True}
    _spend(settings)
    try:
        try:
            resp = _stream(access_token(settings), body)
        except urllib.error.HTTPError as e:
            if e.code != 401:
                raise
            resp = _stream(access_token(settings, force=True), body)
        with resp:
            return _read_stream(resp)
    except urllib.error.HTTPError as e:
        try:
            error = json.loads(e.read())
            code = (error.get('error') or {}).get('code') if isinstance(error.get('error'), dict) else None
        except (ValueError, AttributeError):
            code = None
        raise ChatGPTError(f'model returned {e.code}' + (f': {code}' if code else '')) from None
    except (urllib.error.URLError, TimeoutError) as e:
        raise ChatGPTError(f'model unreachable: {getattr(e, "reason", e)}') from None
    except (http.client.HTTPException, ConnectionError) as e:
        raise ChatGPTError(f'model stream interrupted: {type(e).__name__}') from None


def _read_stream(resp) -> tuple[str, dict]:
    parts = []
    for raw in resp:
        line = raw.decode().strip()
        if not line.startswith('data:'):
            continue
        try:
            event = json.loads(line[5:])
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        kind = event.get('type')
        response = event.get('response') if isinstance(event.get('response'), dict) else {}
        if kind == 'response.output_text.delta':
            parts.append(str(event.get('delta', '')))
        elif kind == 'response.completed':
            return ''.join(parts), response.get('usage') or {}
        elif kind == 'response.incomplete':
            reason = (response.get('incomplete_details') or {}).get('reason') or 'unknown'
            raise ChatGPTError(f'model stopped early: {reason}')
        elif kind in ('response.failed', 'error'):
            error = response.get('error') or event.get('error') or event
            code = error.get('code') if isinstance(error, dict) else error
            raise ChatGPTError(f"model failed: {code or 'unknown_error'}")
    raise ChatGPTError('model stream ended before the response completed')
