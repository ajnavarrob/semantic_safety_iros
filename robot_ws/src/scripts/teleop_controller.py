#!/usr/bin/env python3
"""
Xbox Controller Teleop Node using 'inputs' library.
Works headlessly without display server.

Controller mapping (Xbox):
- Left Stick Y: Forward/Backward (vx)
- Left Stick X: Strafe Left/Right (vy)  
- Right Stick X: Rotation (vyaw)
- A Button: Space (start/save/stop)
- B Button: 'r' (toggle realtime safety filter)
- X Button: 'p' (toggle predictive safety filter)
- Y Button: 'd' (deal parameter)
- Start Button: 'q' (quit)
- D-Pad: Number keys 1-4 for wn parameter
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Int32
import threading
import inputs


class TeleopControllerNode(Node):
    def __init__(self):
        super().__init__('teleop_controller')
        
        # Declare parameters
        self.declare_parameter('vel_max_x_fwd', 0.9)
        self.declare_parameter('vel_max_x_bwd', 0.9)
        self.declare_parameter('vel_max_y', 0.9)
        self.declare_parameter('vel_max_yaw', 0.8)
        self.declare_parameter('deadzone', 0.1)
        self.declare_parameter('publish_rate', 100.0)
        
        self.vel_max_x_fwd = self.get_parameter('vel_max_x_fwd').value
        self.vel_max_x_bwd = self.get_parameter('vel_max_x_bwd').value
        self.vel_max_y = self.get_parameter('vel_max_y').value
        self.vel_max_yaw = self.get_parameter('vel_max_yaw').value
        self.deadzone = self.get_parameter('deadzone').value
        publish_rate = self.get_parameter('publish_rate').value
        
        self.get_logger().info(
            f'Velocity bounds: x_fwd={self.vel_max_x_fwd:.2f}, x_bwd={self.vel_max_x_bwd:.2f}, '
            f'y={self.vel_max_y:.2f}, yaw={self.vel_max_yaw:.2f}'
        )
        
        # Publishers
        self.twist_pub = self.create_publisher(Twist, 'u_des', 1)
        self.key_pub = self.create_publisher(Int32, 'key_press', 1)
        
        # Controller state (normalized -1 to 1)
        self.axis_lx = 0.0  # Left stick X
        self.axis_ly = 0.0  # Left stick Y
        self.axis_rx = 0.0  # Right stick X
        self.axis_ry = 0.0  # Right stick Y
        self.lock = threading.Lock()
        
        # Button state for edge detection
        self.prev_buttons = {}
        
        # Pending key for next publish cycle (0 = no key)
        self.pending_key = 0
        
        # Timer for publishing at constant rate (matches C++ 100Hz)
        self.timer = self.create_timer(1.0 / publish_rate, self.publish_callback)
        
        # Start controller input thread
        self.running = True
        self.input_thread = threading.Thread(target=self.read_controller, daemon=True)
        self.input_thread.start()
        
        self.get_logger().info('Xbox controller teleop started. Waiting for controller...')
    
    def normalize_axis(self, value: int, max_val: int = 32768) -> float:
        """Normalize axis value to -1 to 1 range with deadzone."""
        normalized = value / max_val
        if abs(normalized) < self.deadzone:
            return 0.0
        return normalized
    
    def read_controller(self):
        """Background thread to read controller events."""
        while self.running:
            try:
                events = inputs.get_gamepad()
                for event in events:
                    self.handle_event(event)
            except inputs.UnpluggedError:
                self.get_logger().warn('Controller disconnected. Waiting...')
                import time
                time.sleep(1.0)
            except Exception as e:
                self.get_logger().error(f'Controller error: {e}')
                import time
                time.sleep(0.1)
    
    def handle_event(self, event):
        """Handle a single controller event."""
        code = event.code
        state = event.state
        
        with self.lock:
            # Axis events (sticks)
            if code == 'ABS_X':  # Left stick X
                self.axis_lx = self.normalize_axis(state)
            elif code == 'ABS_Y':  # Left stick Y (inverted)
                self.axis_ly = -self.normalize_axis(state)
            elif code == 'ABS_RX':  # Right stick X
                self.axis_rx = self.normalize_axis(state)
            elif code == 'ABS_RY':  # Right stick Y
                self.axis_ry = -self.normalize_axis(state)
            
            # Button events (only trigger on press, not release)
            # Queue key for next publish cycle instead of publishing directly
            elif code == 'BTN_SOUTH' and state == 1:  # A button -> Space
                self.pending_key = ord(' ')
                self.get_logger().info('A pressed -> Space')
            elif code == 'BTN_EAST' and state == 1:  # B button -> 'r'
                self.pending_key = ord('r')
                self.get_logger().info('B pressed -> r (realtime SF toggle)')
            elif code == 'BTN_WEST' and state == 1:  # X button -> 'p'  
                self.pending_key = ord('p')
                self.get_logger().info('X pressed -> p (predictive SF toggle)')
            elif code == 'BTN_NORTH' and state == 1:  # Y button -> 'd'
                self.pending_key = ord('d')
                self.get_logger().info('Y pressed -> d (deal parameter)')
            elif code == 'BTN_START' and state == 1:  # Start button -> quit
                self.get_logger().info('Start pressed -> Shutting down')
                self.running = False
                rclpy.shutdown()
            
            # D-Pad for wn parameter (1-6)
            elif code == 'ABS_HAT0X':  # D-pad left/right
                if state == -1:  # Left
                    self.pending_key = ord('1')
                    self.get_logger().info('D-Pad Left -> 1')
                elif state == 1:  # Right
                    self.pending_key = ord('2')
                    self.get_logger().info('D-Pad Right -> 2')
            elif code == 'ABS_HAT0Y':  # D-pad up/down
                if state == -1:  # Up
                    self.pending_key = ord('3')
                    self.get_logger().info('D-Pad Up -> 3')
                elif state == 1:  # Down
                    self.pending_key = ord('4')
                    self.get_logger().info('D-Pad Down -> 4')
            
            # Bumpers for 5-6
            elif code == 'BTN_TL' and state == 1:  # Left bumper
                self.pending_key = ord('5')
                self.get_logger().info('LB pressed -> 5')
            elif code == 'BTN_TR' and state == 1:  # Right bumper
                self.pending_key = ord('6')
                self.get_logger().info('RB pressed -> 6')
    
    def publish_key(self, key_code: int):
        """Publish a key press."""
        msg = Int32()
        msg.data = key_code
        self.key_pub.publish(msg)
    
    def publish_callback(self):
        """Timer callback to publish velocity commands and key_press (for timing)."""
        with self.lock:
            twist = Twist()
            
            # Left stick Y -> forward/backward
            if self.axis_ly > 0:
                twist.linear.x = self.axis_ly * self.vel_max_x_fwd
            else:
                twist.linear.x = self.axis_ly * self.vel_max_x_bwd
            
            # Left stick X -> strafe
            twist.linear.y = self.axis_lx * self.vel_max_y
            
            # Right stick X -> rotation
            twist.angular.z = self.axis_rx * self.vel_max_yaw
            
            self.twist_pub.publish(twist)
            
            # Publish pending key or ERR (-1) to maintain constant callback rate
            # This matches C++ teleop behavior for data logging timing
            key_msg = Int32()
            if self.pending_key != 0:
                key_msg.data = self.pending_key
                self.pending_key = 0  # Clear after publishing
            else:
                key_msg.data = -1  # ERR from ncurses
            self.key_pub.publish(key_msg)


def main(args=None):
    rclpy.init(args=args)
    node = TeleopControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.running = False
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
