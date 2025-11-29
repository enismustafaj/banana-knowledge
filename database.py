import redis
import json

r = redis.Redis(host='localhost', port=6379, decode_responses=True)

def add_step(step, workflow_name):
    json_payload = json.dumps(step)

    r.rpush(f"workflows:{workflow_name}", json_payload)

def get_steps(workflow_name):
    items = r.lrange(f"workflows:{workflow_name}", 0, -1)
    steps = [json.loads(item) for item in items]

    return steps

def delete_steps(workflow_name):
    r.delete(f"workflows:{workflow_name}")