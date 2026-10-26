"""Sign in with ChatGPT against a fake OpenAI: authorize, token, JWKS, revocation and Responses on one local server."""
import json
import os
import stat
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlencode, urlsplit

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from optimizer import chatgpt, explain
from optimizer.app import app, settings

ISSUED_CLIENT = 'oaiapp_test'
ADMIN = 'admin-secret'
KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)


class FakeOpenAI(BaseHTTPRequestHandler):
    """Records every request; behaviour is steered through the class attributes the fixture resets."""
    calls = []
    nonce = None
    token_responses = []
    responses_status = 200
    revoke_status = 200
    stream = []

    def log_message(self, *_args):
        pass

    def reply(self, status, body, content_type='application/json'):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # noqa: N802
        FakeOpenAI.calls.append(('GET', self.path, None, dict(self.headers)))
        if self.path == '/jwks':
            jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(KEY.public_key()))
            self.reply(200, {'keys': [{**jwk, 'kid': 'k1', 'use': 'sig', 'alg': 'RS256'}]})
        else:
            self.reply(404, {})

    def do_POST(self):  # noqa: N802
        body = self.rfile.read(int(self.headers.get('Content-Length', 0))).decode()
        FakeOpenAI.calls.append(('POST', self.path, body, dict(self.headers)))
        if self.path == '/token':
            status, response = FakeOpenAI.token_responses.pop(0)
            self.reply(status, response)
        elif self.path == '/revoke':
            self.reply(FakeOpenAI.revoke_status, b'')
        elif self.path == '/responses':
            if FakeOpenAI.responses_status != 200:
                self.reply(FakeOpenAI.responses_status, {'error': {'code': 'subscription_sharing_usage_limit_exceeded'}})
                return
            events = ''.join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in FakeOpenAI.stream)
            self.reply(200, events.encode(), 'text/event-stream')
        else:
            self.reply(404, {})


def id_token(nonce, sub='user-1', aud=ISSUED_CLIENT, iss=None, key=KEY):
    claims = {'iss': iss or chatgpt.ISSUER, 'aud': aud, 'sub': sub, 'nonce': nonce, 'email': 'owner@example.com',
              'iat': int(time.time()), 'exp': int(time.time()) + 3600}
    return jwt.encode(claims, key, algorithm='RS256', headers={'kid': 'k1'})


def tokens(nonce=None, access='access-1', refresh='refresh-1', scope=chatgpt.SCOPES, **extra):
    body = {'access_token': access, 'refresh_token': refresh, 'token_type': 'Bearer', 'expires_in': 3600, 'scope': scope, **extra}
    if nonce is not None:
        body['id_token'] = id_token(nonce)
    return 200, body


@pytest.fixture
def openai(monkeypatch, tmp_path):
    server = ThreadingHTTPServer(('127.0.0.1', 0), FakeOpenAI)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f'http://127.0.0.1:{server.server_port}'
    monkeypatch.setattr(chatgpt, 'TOKEN_URL', base + '/token')
    monkeypatch.setattr(chatgpt, 'JWKS_URL', base + '/jwks')
    monkeypatch.setattr(chatgpt, 'REVOKE_URL', base + '/revoke')
    monkeypatch.setattr(chatgpt, 'RESPONSES_URL', base + '/responses')
    monkeypatch.setattr(settings, 'chatgpt_credentials', str(tmp_path / 'chatgpt.json'))
    monkeypatch.setattr(settings, 'admin_token', ADMIN)
    monkeypatch.setattr(settings, 'azure_openai_endpoint', None)
    monkeypatch.setattr(settings, 'azure_openai_key', None)
    FakeOpenAI.calls, FakeOpenAI.token_responses, FakeOpenAI.responses_status, FakeOpenAI.revoke_status = [], [], 200, 200
    FakeOpenAI.stream = [{'type': 'response.output_text.delta', 'delta': 'Charges '},
                         {'type': 'response.output_text.delta', 'delta': 'at noon.'},
                         {'type': 'response.completed', 'response': {'usage': {'total_tokens': 42}}}]
    yield FakeOpenAI
    server.shutdown()


def callback_for(auth_url, **override):
    query = parse_qs(urlsplit(auth_url).query)
    params = {'code': 'code-1', 'state': query['state'][0], 'client_id': ISSUED_CLIENT, 'scope': chatgpt.SCOPES, **override}
    return query, f'{chatgpt.REDIRECT_URI}?{urlencode(params)}'


def sign_in(fake):
    login = chatgpt.start_login(settings)
    query, url = callback_for(login['auth_url'])
    fake.token_responses.append(tokens(query['nonce'][0]))
    return chatgpt.complete_login(settings, url)


def saved():
    with open(settings.chatgpt_credentials) as f:
        return json.load(f)


def test_first_login_registers_dynamically_with_pkce_and_host_id(openai):
    query = parse_qs(urlsplit(chatgpt.start_login(settings)['auth_url']).query)

    assert query['client_id'] == ['dynamic_agent_client']
    assert query['agent_name_hint'] == [chatgpt.AGENT_NAME]
    assert query['redirect_uri'] == ['http://127.0.0.1:1455/auth/callback']
    assert query['resource'] == ['https://api.openai.com/v1']
    assert set(query['scope'][0].split()) == {'openid', 'profile', 'email', 'offline_access', 'resource.invoke', 'chatgpt.tokens.use.direct'}
    assert query['code_challenge_method'] == ['S256']
    assert query['ext_agent_host_id'][0].startswith('urn:uuid:')


def test_completed_login_stores_the_issued_client_and_tokens_owner_only(openai):
    status = sign_in(openai)

    assert status == {'connected': True, 'email': 'owner@example.com', 'expires_at': pytest.approx(time.time() + 3600, abs=5)}
    exchange = parse_qs(openai.calls[-1][2] if openai.calls[-1][1] == '/token' else next(c[2] for c in openai.calls if c[1] == '/token'))
    assert exchange['client_id'] == [ISSUED_CLIENT]
    assert exchange['grant_type'] == ['authorization_code']
    assert exchange['code_verifier'][0]
    assert exchange['resource'] == ['https://api.openai.com/v1']
    record = saved()
    assert record['client_id'] == ISSUED_CLIENT
    assert record['refresh_token'] == 'refresh-1'
    assert 'pending' not in record
    assert stat.S_IMODE(os.stat(settings.chatgpt_credentials).st_mode) == 0o600


def test_host_id_survives_logins(openai):
    sign_in(openai)
    host = saved()['ext_agent_host_id']
    query = parse_qs(urlsplit(chatgpt.start_login(settings)['auth_url']).query)

    assert query['ext_agent_host_id'] == [host]
    assert query['client_id'] == [ISSUED_CLIENT]
    assert 'agent_name_hint' not in query


@pytest.mark.parametrize('override, message', [
    ({'state': 'forged'}, 'does not belong to the pending sign-in'),
    ({'error': 'access_denied'}, 'declined'),
    ({'client_id': ''}, 'registration was incomplete'),
])
def test_bad_callbacks_are_refused_without_a_token_exchange(openai, override, message):
    _query, url = callback_for(chatgpt.start_login(settings)['auth_url'], **override)

    with pytest.raises(chatgpt.ChatGPTError, match=message):
        chatgpt.complete_login(settings, url)
    assert not [c for c in openai.calls if c[1] == '/token']


def test_login_without_plan_permission_is_refused(openai):
    query, url = callback_for(chatgpt.start_login(settings)['auth_url'])
    openai.token_responses.append(tokens(query['nonce'][0], scope='openid profile email offline_access'))

    with pytest.raises(chatgpt.ChatGPTError, match='ChatGPT plan usage was not granted'):
        chatgpt.complete_login(settings, url)
    assert not chatgpt.connected(settings)


def test_id_token_with_wrong_nonce_is_refused(openai):
    _query, url = callback_for(chatgpt.start_login(settings)['auth_url'])
    openai.token_responses.append(tokens('other-nonce'))

    with pytest.raises(chatgpt.ChatGPTError, match='ID token'):
        chatgpt.complete_login(settings, url)
    assert not chatgpt.connected(settings)


def test_id_token_signed_by_another_key_is_refused(openai):
    query, url = callback_for(chatgpt.start_login(settings)['auth_url'])
    other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    _status, body = tokens()
    openai.token_responses.append((200, {**body, 'id_token': id_token(query['nonce'][0], key=other)}))

    with pytest.raises(chatgpt.ChatGPTError, match='ID token'):
        chatgpt.complete_login(settings, url)


def test_expired_login_is_refused(openai, monkeypatch):
    login = chatgpt.start_login(settings)
    _query, url = callback_for(login['auth_url'])
    monkeypatch.setattr(chatgpt.time, 'time', lambda: login['expires_at'] + 1)

    with pytest.raises(chatgpt.ChatGPTError, match='expired'):
        chatgpt.complete_login(settings, url)


def test_fresh_token_is_used_without_refresh(openai):
    sign_in(openai)

    assert chatgpt.access_token(settings) == 'access-1'
    assert len([c for c in openai.calls if c[1] == '/token']) == 1


def test_token_near_expiry_is_refreshed_and_rotation_saved(openai, monkeypatch):
    sign_in(openai)
    openai.token_responses.append(tokens(access='access-2', refresh='refresh-2'))
    monkeypatch.setattr(chatgpt.time, 'time', lambda: saved()['expires_at'] - 60)

    assert chatgpt.access_token(settings) == 'access-2'
    refresh = parse_qs(openai.calls[-1][2])
    assert refresh == {'grant_type': ['refresh_token'], 'client_id': [ISSUED_CLIENT], 'refresh_token': ['refresh-1'],
                       'resource': ['https://api.openai.com/v1']}
    assert saved()['refresh_token'] == 'refresh-2'


@pytest.mark.parametrize('code', ['invalid_grant', 'refresh_token_reused', 'refresh_token_expired'])
def test_rejected_refresh_disconnects(openai, code):
    sign_in(openai)
    openai.token_responses.append((400, {'error': code}))

    with pytest.raises(chatgpt.ChatGPTError, match='Sign in again'):
        chatgpt.access_token(settings, force=True)
    assert not chatgpt.connected(settings)
    assert saved()['client_id'] == ISSUED_CLIENT


def test_refresh_outage_keeps_the_credentials(openai):
    sign_in(openai)
    openai.token_responses.append((503, {'error': 'temporarily_unavailable'}))

    with pytest.raises(chatgpt.ChatGPTError):
        chatgpt.access_token(settings, force=True)
    assert chatgpt.connected(settings)


def test_logout_revokes_and_clears_tokens_but_keeps_the_registration(openai):
    sign_in(openai)

    assert chatgpt.logout(settings) == {'connected': False}
    revoke = parse_qs(openai.calls[-1][2])
    assert revoke == {'token': ['refresh-1'], 'token_type_hint': ['refresh_token'], 'client_id': [ISSUED_CLIENT]}
    record = saved()
    assert 'refresh_token' not in record and 'access_token' not in record
    assert record['client_id'] == ISSUED_CLIENT


def test_respond_streams_with_the_plan_token(openai):
    sign_in(openai)

    text, usage = chatgpt.respond(settings, 'be brief', 'explain this')

    assert (text, usage) == ('Charges at noon.', {'total_tokens': 42})
    method, path, body, headers = openai.calls[-1]
    request = json.loads(body)
    assert headers['Authorization'] == 'Bearer access-1'
    assert request['store'] is False and request['stream'] is True
    assert request['instructions'] == 'be brief'
    assert request['input'] == [{'role': 'user', 'content': 'explain this'}]
    assert not {'temperature', 'max_output_tokens', 'prompt_cache_retention'} & request.keys()


def test_respond_without_completion_fails(openai):
    sign_in(openai)
    openai.stream = [{'type': 'response.output_text.delta', 'delta': 'half'}]

    with pytest.raises(chatgpt.ChatGPTError, match='ended before'):
        chatgpt.respond(settings, 'be brief', 'explain this')


def test_respond_reports_the_failure_code(openai):
    sign_in(openai)
    openai.stream = [{'type': 'response.failed', 'response': {'error': {'code': 'subscription_sharing_usage_limit_exceeded'}}}]

    with pytest.raises(chatgpt.ChatGPTError, match='subscription_sharing_usage_limit_exceeded'):
        chatgpt.respond(settings, 'be brief', 'explain this')


def test_respond_reports_admission_errors(openai):
    sign_in(openai)
    openai.responses_status = 429

    with pytest.raises(chatgpt.ChatGPTError, match='429'):
        chatgpt.respond(settings, 'be brief', 'explain this')


def test_explain_uses_chatgpt_when_azure_is_unset(openai, payload):
    sign_in(openai)

    response = app.test_client().post('/optimize/explain', json={**payload, 'language': 'en'})

    assert response.status_code == 200
    assert response.json == {'explanation': 'Charges at noon.'}
    request = json.loads(openai.calls[-1][2])
    assert request['model'] == settings.chatgpt_model
    assert request['instructions'].startswith(explain.SYSTEM_PROMPT)


def test_explain_before_sign_in_is_503(openai, payload):
    response = app.test_client().post('/optimize/explain', json=payload)

    assert response.status_code == 503


@pytest.mark.parametrize('headers', [{}, {'X-Admin-Token': 'wrong'}])
def test_admin_endpoints_require_the_admin_token(openai, headers):
    client = app.test_client()

    assert client.get('/chatgpt/status', headers=headers).status_code == 401
    assert client.post('/chatgpt/login', headers=headers).status_code == 401


def test_admin_endpoints_are_off_without_an_admin_token(openai, monkeypatch):
    monkeypatch.setattr(settings, 'admin_token', None)

    assert app.test_client().get('/chatgpt/status', headers={'X-Admin-Token': ADMIN}).status_code == 404


def test_admin_sign_in_round_trip(openai):
    client = app.test_client()
    headers = {'X-Admin-Token': ADMIN}

    login = client.post('/chatgpt/login', headers=headers).json
    query, url = callback_for(login['auth_url'])
    openai.token_responses.append(tokens(query['nonce'][0]))
    done = client.post('/chatgpt/callback', headers=headers, json={'url': url})

    assert done.status_code == 200
    assert done.json['connected'] is True
    assert client.get('/chatgpt/status', headers=headers).json['email'] == 'owner@example.com'
    assert 'access-1' not in done.get_data(as_text=True)
    assert client.post('/chatgpt/logout', headers=headers).json['connected'] is False


def test_admin_callback_errors_are_400(openai):
    client = app.test_client()
    headers = {'X-Admin-Token': ADMIN}
    client.post('/chatgpt/login', headers=headers)

    response = client.post('/chatgpt/callback', headers=headers, json={'url': 'http://127.0.0.1:1455/auth/callback?code=x&state=forged'})

    assert response.status_code == 400


def test_relogin_sends_the_retained_id_token_as_hint(openai):
    sign_in(openai)
    query = parse_qs(urlsplit(chatgpt.start_login(settings)['auth_url']).query)

    assert query['id_token_hint'] == [saved()['id_token']]
    assert query['login_hint'] == ['owner@example.com']


def test_callback_for_another_registration_keeps_the_connected_one(openai):
    sign_in(openai)
    _query, url = callback_for(chatgpt.start_login(settings)['auth_url'], client_id='oaiapp_other')

    with pytest.raises(chatgpt.ChatGPTError, match='another ChatGPT registration'):
        chatgpt.complete_login(settings, url)
    assert saved()['client_id'] == ISSUED_CLIENT
    assert chatgpt.access_token(settings) == 'access-1'


def test_rejected_sign_in_code_is_a_400(openai):
    _query, url = callback_for(chatgpt.start_login(settings)['auth_url'])
    openai.token_responses.append((400, {'error': 'invalid_grant'}))

    with pytest.raises(chatgpt.ChatGPTError, match='rejected the sign-in code') as e:
        chatgpt.complete_login(settings, url)
    assert e.value.status == 400


def test_logout_warns_when_revocation_is_unconfirmed(openai):
    sign_in(openai)
    openai.revoke_status = 503

    assert chatgpt.logout(settings) == {'connected': False, 'warning': chatgpt.REVOCATION_UNCONFIRMED}
    assert not chatgpt.connected(settings)


def test_respond_reports_an_incomplete_response(openai):
    sign_in(openai)
    openai.stream = [{'type': 'response.incomplete', 'response': {'incomplete_details': {'reason': 'max_output_tokens'}}}]

    with pytest.raises(chatgpt.ChatGPTError, match='stopped early: max_output_tokens'):
        chatgpt.respond(settings, 'be brief', 'explain this')


def test_explain_with_unreadable_credentials_is_503(openai, payload):
    with open(settings.chatgpt_credentials, 'w') as f:
        f.write('not json')

    response = app.test_client().post('/optimize/explain', json=payload)

    assert response.status_code == 503


def test_explain_after_sign_out_elsewhere_is_an_error_not_a_crash(openai, payload, monkeypatch):
    sign_in(openai)
    # another worker signs out between the configured check and the model call
    monkeypatch.setattr(explain, 'configured', lambda _settings: True)
    chatgpt.logout(settings)

    response = app.test_client().post('/optimize/explain', json=payload)

    assert response.status_code == 502
    assert 'Sign in with ChatGPT first' in response.json['message']


@pytest.mark.parametrize('content', ['[]', '"text"', '42', 'null'])
def test_credentials_that_are_not_an_object_are_503(openai, payload, content):
    with open(settings.chatgpt_credentials, 'w') as f:
        f.write(content)

    assert app.test_client().post('/optimize/explain', json=payload).status_code == 503
    assert app.test_client().get('/chatgpt/status', headers={'X-Admin-Token': ADMIN}).status_code == 503


def test_daily_limit_caps_plan_explanations(openai, payload, monkeypatch):
    sign_in(openai)
    monkeypatch.setattr(settings, 'chatgpt_daily_limit', 2)
    client = app.test_client()

    assert [client.post('/optimize/explain', json=payload).status_code for _ in range(3)] == [200, 200, 429]
    assert len([c for c in openai.calls if c[1] == '/responses']) == 2


def test_daily_limit_resets_the_next_day(openai, payload, monkeypatch):
    sign_in(openai)
    monkeypatch.setattr(settings, 'chatgpt_daily_limit', 1)
    client = app.test_client()
    monkeypatch.setattr(chatgpt, 'today', lambda: '2026-10-26')
    assert [client.post('/optimize/explain', json=payload).status_code for _ in range(2)] == [200, 429]

    monkeypatch.setattr(chatgpt, 'today', lambda: '2026-10-27')
    assert client.post('/optimize/explain', json=payload).status_code == 200


def test_no_daily_limit_when_zero(openai, payload, monkeypatch):
    sign_in(openai)
    monkeypatch.setattr(settings, 'chatgpt_daily_limit', 0)
    client = app.test_client()

    assert {client.post('/optimize/explain', json=payload).status_code for _ in range(3)} == {200}


def test_model_names_the_provider_that_answers(openai, monkeypatch):
    monkeypatch.setattr(settings, 'chatgpt_model', 'gpt-test-plan')
    monkeypatch.setattr(settings, 'azure_openai_deployment', 'azure-deployment')
    assert explain.model(settings) == 'azure-deployment'  # not signed in yet

    sign_in(openai)
    assert explain.model(settings) == 'gpt-test-plan'

    monkeypatch.setattr(settings, 'azure_openai_endpoint', 'https://example.invalid/')
    monkeypatch.setattr(settings, 'azure_openai_key', 'test-key')
    assert explain.model(settings) == 'azure-deployment'
