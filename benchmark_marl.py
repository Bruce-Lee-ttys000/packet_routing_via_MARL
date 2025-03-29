import argparse
from xuance import get_runner


def parse_args():
    parser = argparse.ArgumentParser("Run benchmark results for MARL.")
    # 可以选择 mappo, ippo, maddpg, iddpg, masac, isac, matd3, iac
    parser.add_argument("--method", type=str, default="mappo")
    parser.add_argument("--env", type=str, default="packet_routing")
    parser.add_argument("--env-id", type=str, default="123")
    parser.add_argument("--test", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


if __name__ == '__main__':
    parser = parse_args()
    runner = get_runner(method=parser.method,
                        env=parser.env,
                        env_id=parser.env_id,
                        parser_args=parser,
                        is_test=parser.test)
    runner.benchmark()
