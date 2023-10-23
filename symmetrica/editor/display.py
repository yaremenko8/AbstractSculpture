import os
import sys
import pickle
import moderngl_window as mglw
import numpy as np
from shared_memory_dict import SharedMemoryDict


class Example(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "ModernGL Example"
    window_size = (1280, 720)
    aspect_ratio = 16 / 9
    resizable = True

    # resource_dir = os.path.normpath(os.path.join(__file__, '../data'))


def raymarch(VERTEX_SHADER, FRAGMENT_SHADER):
    class Raymarching(Example):
        gl_version = (3, 3)
        window_size = (500, 500)
        aspect_ratio = 1.0

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.vaos = []

            program = self.ctx.program(
                vertex_shader=VERTEX_SHADER,
                fragment_shader=FRAGMENT_SHADER
            )

            vertex_data = np.array([
                # x,    y,   z,    u,   v
                -1.0, -1.0, 0.0,  0.0, 0.0,
                +1.0, -1.0, 0.0,  1.0, 0.0,
                -1.0, +1.0, 0.0,  0.0, 1.0,
                +1.0, +1.0, 0.0,  1.0, 1.0,
            ]).astype(np.float32)

            content = [(
                self.ctx.buffer(vertex_data),
                '3f 2f',
                'in_vert', 'in_uv'
            )]

            idx_data = np.array([
                0, 1, 2,
                1, 2, 3
            ]).astype(np.int32)

            idx_buffer = self.ctx.buffer(idx_data)
            self.vao = self.ctx.vertex_array(program, content, idx_buffer)
            self.u_time = program.get("T", 0.0)


        def render(self, time: float, frame_time: float):
            self.u_time.value = time
            self.vao.render()
    Raymarching.run()

if __name__ == "__main__":
    smd = SharedMemoryDict(sys.argv[1], int(sys.argv[2]))
    sys.argv = sys.argv[:1]
    VERTEX_SHADER = str(smd["VERTEX_SHADER"])
    FRAGMENT_SHADER = str(smd["FRAGMENT_SHADER"])
    raymarch(VERTEX_SHADER, FRAGMENT_SHADER)
    del smd
