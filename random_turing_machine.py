"""The original script I wrote for this is lost to time, but I think I should be able to rebuild it quite quickly!"""
import numpy as np
import numpy.random as npr
import cv2
import imageio
import matplotlib
from numba import jit
import os

"""
Controls:
    Q - quit
    S - Save Image
    G - Save Gif
    R - Randomize Turing Machine
    N - Speed Up
    M - Slow Down
"""

# I want a 2D turing machine:
# turing_machine: Tape x State --> Tape x State x Direction
# The set Tape allows for a "blank" as to not overwrite

# I want to generate random Turing machines, so I need a way to randomly generate these functions.
# I can generate a random 3-tensor of shape (tape, state, 3)
# I will do this by generateing three seperate (tape, state) matricies because each will have a different range.

DIRS = np.array([(-1, 0), (0, -1), (1, 0), (0, 1)])
xSIZE = 128
ySIZE = 64
scaleSIZE = 8
ALPHABET = 8
STATES = 64
STEPS_PER = 5
BUFFER_LENGTH = 600
CMAP_STRING = "cividis"

if not os.path.exists("results"):
    os.makedirs("results")

def generate_random_turing(num_tape, num_state):
    """Generates a new turing matrix"""
    kernel = npr.randint(0,1, size = (num_tape, num_tape), dtype=np.uint16)

    tape_matrix = npr.randint(0, num_tape, size=(num_tape, num_state), dtype=np.uint16)

    state_matrix = npr.randint(
        0, num_state, size=(num_tape, num_state), dtype=np.uint16
    )

    dir_matrix = npr.randint(0, 4, size=(num_tape, num_state), dtype=np.uint16)
    return np.stack((tape_matrix, state_matrix, dir_matrix), axis=-1, dtype=np.uint16)


@jit(nopython=True)
def take_step(tape, state, location, turing_matrix, alphabet, dirs, xsize, ysize):
    """Takes one turing step"""
    tape_symbol = tape[location]
    nsymb, nstate, ndir = turing_matrix[(tape_symbol, state)]
    tape[location] = (tape[location] + nsymb) % alphabet

    nlocation = (
        (location[0] + dirs[ndir][0]) % ysize,
        (location[1] + dirs[ndir][1]) % xsize,
    )
    return tape, nstate, nlocation


# Initialize the all parameters
tape = np.zeros((xSIZE, ySIZE), dtype=np.uint16).T
# tape = npr.randint(0, ALPHABET, size=(xSIZE, ySIZE), dtype=np.uint16).T

state = 0
location = (ySIZE // 2, xSIZE // 2)
tur_mach = generate_random_turing(ALPHABET, STATES)

# Generate color map and normalization
cmap = matplotlib.colormaps[CMAP_STRING]
norm = matplotlib.colors.Normalize(vmin=0, vmax=ALPHABET)


rgb_img = (cmap(norm(tape)) * 255).astype(np.uint8)
rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
rgb_img = cv2.resize(
    rgb_img, (scaleSIZE * xSIZE, scaleSIZE * ySIZE), interpolation=cv2.INTER_NEAREST
)

# Initialize window
cv2.namedWindow("TuringMachine", cv2.WINDOW_NORMAL)
i = 0
prev_key = -1
gif_buffer = []
while True:
    for j in range(STEPS_PER):
        tape, state, location = take_step(
            tape,
            state,
            location,
            tur_mach,
            ALPHABET,
            DIRS,
            xSIZE,
            ySIZE,
        )
    # Re-generate color map, and re-size
    rgb_img = (cmap(norm(tape)) * 255).astype(np.uint8)
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    rgb_img = cv2.resize(
        rgb_img, (scaleSIZE * xSIZE, scaleSIZE * ySIZE), interpolation=cv2.INTER_NEAREST
    )

    gif_buffer = gif_buffer[-BUFFER_LENGTH:] + [rgb_img]

    cv2.imshow("TuringMachine", rgb_img)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    elif key == ord("s"):
        i += 1
        cv2.imwrite(f"results/{ALPHABET}_{STATES}_{i}.png", rgb_img)
    elif key == ord("g"):
        imageio.mimsave(
            f"results/video{ALPHABET}_{STATES}_{i}.gif",
            gif_buffer,
            fps=60,
            format="GIF-PIL",
        )
        i += 1
    elif key == ord("r"):
        tur_mach = generate_random_turing(ALPHABET, STATES)
    elif key == ord("n"):
        STEPS_PER += 1
        print(f"Speed: {STEPS_PER}")
    elif key == ord("m"):
        STEPS_PER = max(STEPS_PER - 1, 0)
        print(f"Speed: {STEPS_PER}")


i += 1
cv2.destroyAllWindows()
cv2.imwrite(f"results/final_{ALPHABET}_{STATES}_{i}.png", rgb_img)
