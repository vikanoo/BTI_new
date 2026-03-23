"""
Deploy BTI_NEW workflow to n8n with real API keys injected.
Keys are stored in config.json (gitignored) and never committed.
"""
import json
import urllib.request
import urllib.error
import sys

with open("config.json", encoding="utf-8") as f:
    cfg = json.load(f)

with open("Workflows/BTI_NEW.json", encoding="utf-8") as f:
    wf_text = f.read()

# Inject real keys
wf_text = wf_text.replace("SUPABASE_SERVICE_ROLE_KEY", cfg["supabase_service_role_key"])
wf_text = wf_text.replace("OPENAI_API_KEY", cfg["openai_api_key"])

wf = json.loads(wf_text)

payload = json.dumps({
    "name": wf.get("name", "BTI_NEW"),
    "nodes": wf["nodes"],
    "connections": wf["connections"],
    "settings": wf.get("settings", {}),
    "staticData": wf.get("staticData")
}, ensure_ascii=True).encode("utf-8")

req = urllib.request.Request(
    f"{cfg['n8n_api_url']}/workflows/{cfg['workflow_id']}",
    data=payload,
    method="PUT",
    headers={
        "X-N8N-API-KEY": cfg["n8n_api_key"],
        "Content-Type": "application/json"
    }
)

try:
    with urllib.request.urlopen(req) as resp:
        print(f"OK {resp.status} — workflow deployed with real keys")
except urllib.error.HTTPError as e:
    print(f"ERROR {e.code}: {e.read().decode()[:500]}")
    sys.exit(1)
