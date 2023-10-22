import symmetrica
from symmetrica import shape, isometry
import sympy as sp
import re

T = sp.symbols('T')
I = sp.eye(3)
e1 = I[:, 0] #* (0.55 + 0.45 * sp.sin(T))
e2 = I[:, 1]
e3 = I[:, 2]
O = I[:, 0] * 0.0
p = sp.MatrixSymbol('p', 3, 1)

sphere = shape.core.Sphere(1, center=e1)
triangle = shape.core.Triangle(e1, e2, e3)
simplex = shape.core.Simplex(e1, e2, e3, O)
spiky = isometry.IcosahedralGroup() @ simplex

spiky.display()
spiky.display()
#exit()
input()

expr = spiky(p)

def expression_to_glsl(expr):
    #expr = sp.glsl_code(expr) # It appears that Max, Min get converted to pure C, which takes a ridiculous ammount of time and produces a massive expression
    #expr = sp.expand_power_exp(expr)
    expr = str(expr)
    for Ext, ext in [("Max", "max"), ("Min", "min")]:
        while Ext in expr:
            Ext_idx = expr.find(Ext)
            i = Ext_idx + 3
            nesting_counter = 0
            got_comma = False
            forbidden_commas_present = False
            while nesting_counter != -1:
                i += 1
                if expr[i] == '(' or expr[i] == '[':
                    nesting_counter += 1
                elif expr[i] == ')' or expr[i] == ']':
                    nesting_counter -= 1
                elif expr[i] == ',' and nesting_counter == 0:
                    if not got_comma:
                        got_comma = True
                        comma_index = i
                    else:
                        forbidden_commas_present = True
            right_prt_idx = i
            if forbidden_commas_present:
                expr = expr[:right_prt_idx] + ')' + expr[right_prt_idx:]
                expr = expr[:comma_index + 1] + Ext + '(' + expr[comma_index + 1:]
            expr = expr[:Ext_idx] + ext + expr[Ext_idx + 3:]
    #expr = expr.replace("Max", "vmax")
    #expr = expr.replace("Min", "vmin")
    expr = expr.replace("[0, 0]", ".x")
    expr = expr.replace("[1, 0]", ".y")
    expr = expr.replace("[2, 0]", ".z")
    #expr = expr.replace("[0][0]", ".x")
    #expr = expr.replace("[1][0]", ".y")
    #expr = expr.replace("[2][0]", ".z")
    expr = re.sub(r"([A-Za-z0-9_\.]+)\*\*([0-9\.]+)", r"pow(\1, \2.0)", expr)
    expr = re.sub(r"Piecewise\(\(1,(.*?)\), \(0, True\)\)", r"float(\1)", expr)
    expr = re.sub(r"(\*\*|/)([0-9\.]+)", r"\1(\2.0)", expr)
    expr = re.sub(r"([0-9]\.[0-9])\.0", r"\1", expr)
    exponentiations = []
    stack = []
    argument_stacklevels = {}
    exponentiations = []
    for i, s in enumerate(expr): # This will fail if thre is an expression of the kind a ** b ** c
        if s == "(":
            stack.append(i)
        elif s == ")":
            if expr[i + 1: i + 3] == '**':
                argument_stacklevels[len(stack) - 2] = slice(stack[-1], i + 1)
            elif len(stack) - 2 in argument_stacklevels:
                exponentiations.append((len(stack),
                                        expr[argument_stacklevels[len(stack) - 2]],
                                        expr[stack[-1]: i + 1]))
                del argument_stacklevels[len(stack) - 2]
            stack.pop(-1)
    exponentiations.sort()
    for _, left, right in exponentiations:
        expr = expr.replace(f"{left}**{right}", f"pow({left}, {right})")

    return expr


import numpy as np
import os
import moderngl_window as mglw


class Example(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "ModernGL Example"
    window_size = (1280, 720)
    aspect_ratio = 16 / 9
    resizable = True

    # resource_dir = os.path.normpath(os.path.join(__file__, '../data'))



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


VERTEX_SHADER = '''
#version 430

in vec3 in_vert;
in vec2 in_uv;
out vec2 v_uv;
void main()
{
    gl_Position = vec4(in_vert.xyz, 1.0);
    v_uv = in_uv;
}
'''

FRAGMENT_SHADER = '''
#version 430

#define FAR 160.0
#define MARCHING_MINSTEP 0
#define MARCHING_STEPS 512
#define MARCHING_CLAMP 0.000001
#define NRM_OFS 0.0001
#define AO_OFS 0.00001
#define PI 3.141592
#define FOG_DIST 2.5
#define FOG_DENSITY 0.32
#define FOG_COLOR vec3(0.25, 0.27, 0.82)

layout(location=0) uniform float T;

// in vec2 v_uv: screen space coordniate
in vec2 v_uv;

// out color
out vec4 out_color;

float vmax(float a, float b, float c, float e, float f)
{
    return max(a, max(b, max(c, max(e, f))));
}

// p: sample position
// r: rotation in Euler angles (radian)
vec3 rotate(vec3 p, vec3 r)
{
    vec3 c = cos(r);
    vec3 s = sin(r);
    mat3 rx = mat3(
        1, 0, 0,
        0, c.x, -s.x,
        0, s.x, c.s
    );
    mat3 ry = mat3(
        c.y, 0, s.y,
        0, 1, 0,
        -s.y, 0, c.y
    );
    mat3 rz = mat3(
        c.z, -s.z, 0,
        s.z, c.z, 0,
        0, 0, 1
    );
    return rz * ry * rx * p;
}

// p: sample position
// t: tiling distance
vec3 tile(vec3 p, vec3 t)
{
    return mod(p, t) - 0.5 * t;
}

// p: sample position
// r: radius
float sphere(vec3 p, float r)
{
    return length(p) - r;
}

// p: sample position
// b: width, height, length (scalar along x, y, z axis)
float box(vec3 p, vec3 b)
{
    return length(max(abs(p) - b, 0.0));
}

// c.x, c.y: offset
// c.z: radius
float cylinder(vec3 p, vec3 c)
{
    return length(p.xz - c.xy) - c.z;
}

// a, b: capsule position from - to
// r: radius r
float capsule(vec3 p, vec3 a, vec3 b, float r)
{
    vec3 dp = p - a;
    vec3 db = b - a;
    float h = clamp(dot(dp, db) / dot(db, db), 0.0, 1.0);
    return length(dp - db * h) - r;
}

// p: sample position
// c: cylinder c
// b: box b
float clamp_cylinder(vec3 p, vec3 c, vec3 b)
{
    return max(cylinder(p, c), box(p, b));
}
// a: primitive a
// b: primitive b
// k: blending amount
float blend(float a, float b, float k)
{
    float h = clamp(0.5 + 0.5 * (a - b) / k, 0.0, 1.0);
    return mix(a, b, h) - k * h * (1.0 - h);
}

float displace(vec3 p, float m, float s)
{
    return sin(p.x * m) * sin(p.y * m) * sin(p.z * m) * s;
}

// world
float sample_world(vec3 p, inout vec3 c)
{
    vec3 b_left_pos = p - vec3(-0.8, -0.25, 0.0);
    b_left_pos = rotate(b_left_pos, vec3(T, 0.0, 0.0));
    float d_box_left = box(b_left_pos, vec3(0.4));

    vec3 b_right_pos = p - vec3(+0.8, -0.25, 0.0);
    b_right_pos = rotate(b_right_pos, vec3(0.0, 0.0, T));
    float d_box_right = box(b_right_pos, vec3(0.4));

    vec3 b_up_pos = p - vec3(0.0, 1.05, 0.0);
    b_up_pos = rotate(b_up_pos, vec3(0.0, T, 0.0));
    float d_box_up = box(b_up_pos, vec3(0.4));

    float d_box = FAR;
    d_box = min(d_box, d_box_left);
    d_box = min(d_box, d_box_right);
    d_box = min(d_box, d_box_up);

    vec3 s_pos = p - vec3(0.0, 0.2, 0.0);
    float d_sphere = sphere(s_pos, 0.65);

    float result = blend(d_sphere, d_box, 0.3);

    if (result < FAR)
    {
        c.x = 0.9;
        c.y = 0.15;
        c.z = 0.15;
    }
    p.xy = vec2(cos(T) * p.x + sin(T) * p.y, -sin(T) * p.x + cos(T) * p.y);
    p.xz = vec2(cos(T / 3.14) * p.x + sin(T / 3.14) * p.z, -sin(T / 3.14) * p.x + cos(T / 3.14) * p.z);
    return ''' + expression_to_glsl(expr) + ''';//result;
    
}

// o: origin
// r: ray
// c: color
float raymarch(vec3 o, vec3 r, inout vec3 c)
{
    float t = 0.0;
    vec3 p = vec3(0);
    float d = 0.0;
    for (int i = MARCHING_MINSTEP; i < MARCHING_STEPS; i++)
    {
        p = o + r * t;
        d = sample_world(p, c);
        if (abs(d) < MARCHING_CLAMP)
        {
            return t;
        }
        t += d * 0.6;
    }
    return FAR;
}

// p: sample surface
vec3 norm(vec3 p)
{
    vec2 o = vec2(NRM_OFS, 0.0);
    vec3 dump_c = vec3(0);
    return normalize(vec3(
        sample_world(p + o.xyy, dump_c) - sample_world(p - o.xyy, dump_c),
        sample_world(p + o.yxy, dump_c) - sample_world(p - o.yxy, dump_c),
        sample_world(p + o.yyx, dump_c) - sample_world(p - o.yyx, dump_c)
    ));
}

void main()
{
    // o: origin
    vec3 o = vec3(0.0, 0.5, -6.0);

    // r: ray
    vec3 r = normalize(vec3(v_uv - vec2(0.5, 0.5), 1.001));

    // l: light
    vec3 l = normalize(vec3(-0.5, -0.2, 0.1));

    // c: albedo
    vec3 c = vec3(0.125);
    float d = raymarch(o, r, c);

    // pixel color
    vec3 color = vec3(0);
    if (d < FAR)
    {
        vec3 p = o + r * d;
        vec3 n = norm(p);

        float lambert = dot(n, l);
        lambert = clamp(lambert, 0.1, 1.0);

        #define SPEC_COLOR vec3(0.85, 0.75, 0.5)
        vec3 h = normalize(o + l);
        float ndh = clamp(dot(n, h), 0.0, 1.0);
        float ndv = clamp(dot(n, -o), 0.0, 1.0);
        float spec = pow((ndh + ndv) + 0.01, 64.0) * 0.25;

        color = c * lambert + SPEC_COLOR * spec;
    }

    // add simple fog
    color = mix(FOG_COLOR, color, clamp(pow(FOG_DIST / abs(d), FOG_DENSITY), 0.0, 1.0));

    out_color = vec4(color, 1.0);
}
'''


if __name__ == '__main__':
    Raymarching.run()
