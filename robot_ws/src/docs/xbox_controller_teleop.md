# Xbox Controller Teleop

The `teleop_controller.py` script provides Xbox controller input for robot teleoperation, replacing the keyboard-based `teleOp` executable.

## Requirements

```bash
pip install inputs
```

## Controller Mapping

### Movement (Left Stick)

| Input        | Action        |
| ------------ | ------------- |
| Left Stick ↑ | Move forward  |
| Left Stick ↓ | Move backward |
| Left Stick ← | Strafe left   |
| Left Stick → | Strafe right  |

### Rotation (Right Stick)

| Input         | Action     |
| ------------- | ---------- |
| Right Stick ← | Rotate CCW |
| Right Stick → | Rotate CW  |

### Buttons

| Button | Key Code | Action                                       |
| ------ | -------- | -------------------------------------------- |
| A      | Space    | Start/Save/Stop (increments `space_counter`) |
| B      | 'r'      | Toggle realtime safety filter                |
| X      | 'p'      | Toggle predictive safety filter              |
| Y      | 'd'      | Deal parameter from deck                     |
| Start  | -        | Quit node                                    |

### Parameter Selection (wn)

| Input       | Key Code | wn Value |
| ----------- | -------- | -------- |
| D-Pad Left  | '1'      | 0.5      |
| D-Pad Right | '2'      | 1.0      |
| D-Pad Up    | '3'      | 1.5      |
| D-Pad Down  | '4'      | 2.0      |
| LB          | '5'      | 4.0      |
| RB          | '6'      | 8.0      |

## Usage

```bash
ros2 run unitree_ros2_poisson_simple teleop_controller.py
```

## Parameters

| Parameter       | Default | Description                   |
| --------------- | ------- | ----------------------------- |
| `vel_max_x_fwd` | 0.9     | Max forward velocity (m/s)    |
| `vel_max_x_bwd` | 0.9     | Max backward velocity (m/s)   |
| `vel_max_y`     | 0.9     | Max strafe velocity (m/s)     |
| `vel_max_yaw`   | 0.8     | Max rotation velocity (rad/s) |
| `deadzone`      | 0.1     | Stick deadzone (0-1)          |
| `publish_rate`  | 100.0   | Publish rate in Hz            |

## Topics

| Topic       | Type                  | Direction |
| ----------- | --------------------- | --------- |
| `u_des`     | `geometry_msgs/Twist` | Published |
| `key_press` | `std_msgs/Int32`      | Published |
