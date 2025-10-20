import json
import random
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
import numpy as np
if not hasattr(np, "float_"):
    np.float_ = np.float64

class RLlibHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers["Content-Length"])
        post_data = self.rfile.read(content_length)
        try:
            data = json.loads(post_data.decode("utf-8"))
            action = 0

            # Process observations for each agent
            for agent_id in range(1, 5):  # Agents 1-4
                agent_key = f"agent_{agent_id}"
                processed_obs = {}
                # Get the agent's observation from the raw observation
                agent_obs = data.get(agent_key, {})
                if agent_obs == {}:
                    continue
                processed_obs[agent_key] = { #auction mask isn't bad with all buy/sell house -> writing
                    'action_mask': np.array(agent_obs.get("action_mask", [0, 0, 0, 0, 0, 1, 1]+[0]*116), dtype=np.int8),
                    'action': agent_obs.get("action", 0),
                    'position': agent_obs.get("position", 0),
                    'isPrison': agent_obs.get("isPrison", 0),
                    'money': np.array([agent_obs.get("money", [0])[0] if isinstance(agent_obs.get("money", [0]), list) else agent_obs.get("money", 0)], dtype=np.int32),
                    'owned_properties': np.array(agent_obs.get("owned_properties", [0] * 28), dtype=np.int8),
                    'rent': np.array(agent_obs.get("rent", [0.0] * 40), dtype=np.float32),
                    'houses': np.array(agent_obs.get("houses", [0] * 28), dtype=np.int8),
                    'mortgageds': np.array(agent_obs.get("mortgageds", [0] * 28), dtype=np.int8),
                    'other_position': np.array(agent_obs.get("other_position", [0] * 3), dtype=np.int8),
                    'other_isPrison': np.array(agent_obs.get("other_isPrison", [0] * 3), dtype=np.int8),
                    'other_money': np.array(agent_obs.get("other_money", [0] * 3), dtype=np.int32),
                    'other_owned_properties1': np.array(agent_obs.get("other_owned_properties1", [0] * 28), dtype=np.int8),
                    'other_owned_properties2': np.array(agent_obs.get("other_owned_properties2", [0] * 28), dtype=np.int8),
                    'other_owned_properties3': np.array(agent_obs.get("other_owned_properties3", [0] * 28), dtype=np.int8),
                    'auction_state': np.array(agent_obs.get("auction_state", [0] * 4), dtype=np.float32),
                    'obogp': np.array(agent_obs.get("obogp", [0] * 10), dtype=np.float32),
                    'trade_state': np.array(agent_obs.get("trade_state", [0] * 2), dtype=np.int8)
                }
                #print(processed_obs[agent_key])
                action_mask = agent_obs.get("action_mask", [0, 0, 0, 0, 0, 1, 1]+[0]*116)
                valid_actions = [i for i, mask in enumerate(action_mask) if mask == 1]
                action = random.choice(valid_actions) # randomly chooses a valid action
                #print(action_mask, "->", action)

            response = json.dumps(str(action)).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(response)

        except Exception as e:
            traceback.print_exc()
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            error_msg = {"error": str(e)}
            self.wfile.write(json.dumps(error_msg).encode("utf-8"))


def run(server_class=HTTPServer, handler_class=RLlibHandler, port=8081):
    server_address = ("", port)
    httpd = server_class(server_address, handler_class)
    print(f"Starting Random HTTP server on port {port}...")
    httpd.serve_forever()

if __name__ == "__main__":
    run()