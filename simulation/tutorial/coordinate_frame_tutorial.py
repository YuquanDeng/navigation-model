###############################
# Import
###############################
import os
import git
import magnum as mn
from PIL import Image
import habitat_sim

###############################
# SETUP
###############################

# Choose quiet logging. See src/esp/core/Logging.h
os.environ["HABITAT_SIM_LOG"] = "quiet"

# define path to data directory
dir_path = os.getcwd()
# %cd $dir_path
data_path = os.path.join(dir_path, "../data")

# images will be either displayed in the colab or saved as image files
IS_NOTEBOOK = False
if not IS_NOTEBOOK:
    output_directory = "coordinate_system_tutorial_output/"
    output_path = os.path.join(dir_path, output_directory)
    os.makedirs(output_path, exist_ok=True)

# define some constants and globals the first time we run:
opacity = 1.0
red = mn.Color4(1.0, 0.0, 0.0, opacity)
green = mn.Color4(0.0, 1.0, 0.0, opacity)
blue = mn.Color4(0.0, 0.0, 1.0, opacity)
white = mn.Color4(1.0, 1.0, 1.0, opacity)

origin = mn.Vector3(0.0, 0.0, 0.0)
eye_pos0 = mn.Vector3(2.5, 1.3, 1)
eye_pos1 = mn.Vector3(3.5, 3.0, 4.5)
obj_axes_len = 0.4

if "sim" not in globals():
    global sim
    sim = None
    global sensor_node
    sensor_node = None
    global lr
    lr = None
    global image_counter
    image_counter = 0

###############################
# Utilities
###############################
def create_sim_helper(scene_id):
    global sim
    global sensor_node
    global lr

    # clean-up the current simulator instance if it exists
    if sim != None:
        sim.close()

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_dataset_config_file = os.path.join(
        data_path, "replica_cad/replicaCAD.scene_dataset_config.json"
    )
    sim_cfg.scene_id = scene_id
    sim_cfg.enable_physics = True

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [768, 1024]
    rgb_sensor_spec.position = [0.0, 0.0, 0.0]
    agent_cfg.sensor_specifications = [rgb_sensor_spec]

    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(cfg)

    # This tutorial doesn't involve agent concepts. We want to directly set
    # camera transforms in world-space (the world's coordinate frame). We set
    # the agent transform to identify and then return the sensor node.
    sim.initialize_agent(0)
    agent_node = sim.get_agent(0).body.object
    agent_node.translation = [0.0, 0.0, 0.0]
    agent_node.rotation = mn.Quaternion()
    sensor_node = sim._sensors["color_sensor"]._sensor_object.object

    lr = sim.get_debug_line_render() 
    lr.set_line_width(3)

def show_img(rgb_obs):
    global image_counter

    colors = []
    for row in rgb_obs:
        for rgba in row:
            colors.extend([rgba[0], rgba[1], rgba[2]])

    resolution_x = len(rgb_obs[0])
    resolution_y = len(rgb_obs)

    colors = bytes(colors)
    img = Image.frombytes("RGB", (resolution_x, resolution_y), colors)
    if IS_NOTEBOOK:
        IPython.display.display(img)
    else:
        filepath = f"{output_directory}/{image_counter}.png"
        img.save(filepath)
        print(f"Saved image: {filepath}")
        image_counter += 1

def show_scene(camera_transform):
    sensor_node.transformation = camera_transform
    observations = sim.get_sensor_observations()
    show_img(observations["color_sensor"])

def draw_axes(translation, axis_len=1.0):
    lr = sim.get_debug_line_render()
    # draw axes with x+ = red, y+ = green, z+ = blue
    lr.draw_transformed_line(translation, mn.Vector3(axis_len, 0, 0), red)
    lr.draw_transformed_line(translation, mn.Vector3(0, axis_len, 0), green)
    lr.draw_transformed_line(translation, mn.Vector3(0, 0, axis_len), blue)

def calc_camera_transform(
    eye_translation=mn.Vector3(1, 1, 1), lookat=mn.Vector3(0, 0, 0)
):
    # choose y-up to match Habitat's y-up convention
    camera_up = mn.Vector3(0.0, 1.0, 0.0)
    return mn.Matrix4.look_at(eye_translation, lookat, camera_up)

def main():
    create_sim_helper(scene_id="NONE")
    draw_axes(origin)
    # show_scene(calc_camera_transform(eye_translation=eye_pos0, lookat=origin))
    transform_matrix = calc_camera_transform(eye_translation=(-0.5, -0.5, -0.5), lookat=origin)
    print("-"*100)
    print("rotation_matrix: ", transform_matrix)
    print("-"*100)
    show_scene(transform_matrix)


    return 
    create_sim_helper(
    scene_id=os.path.join(
        data_path, "replica_cad/configs/scenes/v3_sc0_staging_00.scene_instance.json"
    )
    )
    draw_axes(origin)
    camera_transform = calc_camera_transform(eye_translation=eye_pos0, lookat=origin)
    show_scene(camera_transform)

    # Test for axis.

    # draw the previous camera's local axes
    lr.push_transform(camera_transform)
    draw_axes(origin, axis_len=obj_axes_len)
    # draw some approximate edges of the previous camera's frustum
    fx = 0.5
    fy = 0.5
    fz = 0.5
    # lr.draw_transformed_line(origin, mn.Vector3(-fx, 0, 0), white)
    lr.draw_transformed_line(origin, mn.Vector3(0, -fy, 0), white)
    lr.draw_transformed_line(origin, mn.Vector3(0, 0, -fz), white)
    lr.pop_transform()

    # Show the scene from a position slightly offset from the previous camera.
    eye_offset = mn.Vector3(0.5, 0.75, 1.75)
    show_scene(calc_camera_transform(eye_translation=eye_pos0 + eye_offset, lookat=origin))


if __name__ == "__main__":
    main()