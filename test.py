import argparse
import torch
from src.deep_q_network import DeepQNetwork
from src.flappy_bird import FlappyBird
from src.utils import pre_processing


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Flappy Bird""")
    parser.add_argument("--image_size", type=int, default=84, help="The common width and height for all images")
    parser.add_argument("--saved_path", type=str, default="trained_models")

    args = parser.parse_args()
    return args

def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load("{}/flappy_bird".format(args.saved_path), weights_only=False)
    model.eval()
    game_state = FlappyBird()
    image, reward, terminal = game_state.next_frame(0)
    image = pre_processing(image[:game_state.screen_width, :int(game_state.base_y)], args.image_size, args.image_size)
    image = torch.from_numpy(image)

    model = model.to(device)
    image = image.to(device)

    state = torch.cat(tuple(image for _ in range(4)))[None, :, :, :]

    while True:
        prediction = model(state)[0]
        action = torch.argmax(prediction).item()

        next_image, reward, terminal = game_state.next_frame(action)
        next_image = pre_processing(next_image[:game_state.screen_width, :int(game_state.base_y)], args.image_size, args.image_size)
        next_image = torch.from_numpy(next_image)
        next_image = next_image.to(device)
        next_state = torch.cat((state[0, 1:, :, :], next_image))[None, :, :, :]
        state = next_state

if __name__ == '__main__':
    args = get_args()
    test(args)