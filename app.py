#!/usr/bin/env python

import gradio as gr
import numpy as np

from model import Model

DESCRIPTION = '# <a href="https://github.com/kuprel/min-dalle">min(DALL·E)</a>'

model = Model()


def make_grid(images: list[np.ndarray], nrows: int, ncols: int) -> np.ndarray:
    h, w = images[0].shape[:2]
    grid = np.zeros((h * nrows, w * ncols, 3), dtype=np.uint8)
    for i in range(nrows):
        for j in range(ncols):
            index = ncols * i + j
            if index >= len(images):
                break
            grid[h * i:h * (i + 1), w * j:w * (j + 1)] = images[index]
    return grid


def generate(text: str, is_mega: bool, seed: int, nrows: int,
             ncols: int) -> tuple[np.ndarray, list[np.ndarray]]:
    seeds = [seed + i for i in range(nrows * ncols)]
    res = model.generate_images(text, seeds, is_mega)
    grid = make_grid(res, nrows, ncols)
    return grid, res


def set_example_text(example: list) -> dict:
    return gr.Textbox.update(value=example[0])


def main():
    with gr.Blocks(css='style.css') as demo:
        gr.Markdown(DESCRIPTION)

        with gr.Row():
            with gr.Column():
                with gr.Group():
                    text = gr.Textbox(label='Input Text')
                    is_mega = gr.Checkbox(label='Mega', value=True)
                    seed = gr.Slider(0, 100000, value=0, step=1, label='Seed')
                    nrows = gr.Slider(1,
                                      4,
                                      value=2,
                                      step=1,
                                      label='Number of Rows')
                    ncols = gr.Slider(1,
                                      4,
                                      value=2,
                                      step=1,
                                      label='Number of Columns')
                    run_button = gr.Button('Run')

                    with open('samples.txt') as f:
                        samples = [[line.strip()] for line in f.readlines()]
                    examples = gr.Dataset(components=[text], samples=samples)

            with gr.Column():
                with gr.Tabs():
                    with gr.TabItem('Output (Grid View)'):
                        result_grid = gr.Image(show_label=False)
                    with gr.TabItem('Output (Gallery)'):
                        result_gallery = gr.Gallery(show_label=False)

        run_button.click(fn=generate,
                         inputs=[
                             text,
                             is_mega,
                             seed,
                             nrows,
                             ncols,
                         ],
                         outputs=[
                             result_grid,
                             result_gallery,
                         ])
        examples.click(fn=set_example_text,
                       inputs=examples,
                       outputs=examples.components)

    demo.launch(share=False, enable_queue=True)


if __name__ == '__main__':
    main()
