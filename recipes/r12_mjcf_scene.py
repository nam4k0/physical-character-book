#!/usr/bin/env python3
"""レシピ12: 自作シーンを MuJoCo で書いて転がす（第11章）。

    uv run python recipes/r12_mjcf_scene.py
"""

XML = """
<mujoco model="my_scene">
  <option timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="box" pos="0 0 1">
      <geom type="box" size="0.2 0.2 0.2" rgba="1 0.5 0 1"/>
      <freejoint/>
    </body>
  </worldbody>
</mujoco>
"""


def main() -> None:
    import mujoco

    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    for step in range(2000):
        mujoco.mj_step(model, data)
        if step % 500 == 0:
            print(f"step {step}: 箱の高さ = {data.qpos[2]:.3f}")


if __name__ == "__main__":
    main()
