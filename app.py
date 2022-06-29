#!/usr/bin/env python

import gradio as gr
import numpy as np

from model import AppModel

DESCRIPTION = '# <a href="https://github.com/kuprel/min-dalle">min(DALL·E)</a>'


def set_example_text(example: list) -> dict:
    return gr.Textbox.update(value=example[0])


def main():
    model = AppModel()
    with gr.Blocks(css='style.css') as demo:
        gr.Markdown(DESCRIPTION)

        with gr.Row():
            with gr.Column():
                with gr.Group():
                    text = gr.Textbox(label='Input Text')
                    model_name = gr.Radio(choices=['dalle-mini', 'dalle-mega'],
                                          value='dalle-mini',
                                          type='value',
                                          label='Model')
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

        run_button.click(fn=model.run,
                         inputs=[
                             text,
                             model_name,
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
