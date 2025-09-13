import time
import requests

class Alerter:
    def __init__(self, cfg):
        self.enabled = cfg['alerting'].get('enabled', False)
        self.targets = set(cfg['alerting'].get('target_classes', []))
        self.cooldown = cfg['alerting'].get('cooldown_seconds', 30)
        self.webhooks = cfg['alerting'].get('webhooks', [])
        self.last_sent = 0

    def maybe_alert(self, events):
        if not self.enabled or not events:
            return
        now = time.time()
        if now - self.last_sent < self.cooldown:
            return
        payload = {"events": events, "ts": int(now)}
        for wh in self.webhooks:
            try:
                requests.post(wh['url'], json=payload, timeout=2)
            except Exception:
                pass
        self.last_sent = now
