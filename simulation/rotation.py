#!/usr/bin/env python3

import rospy
import tf
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Point, Pose, Quaternion
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import time
import os

ROBOT_MODEL_NAME = "robot"  
CMD_VEL_TOPIC = "/vector/cmd_vel" 
ODOM_TOPIC = "/odom" 

TARGET_ANGULAR_VELOCITY_RAD_S = 0.5 
TARGET_ROTATION_DEG = 90.0
NUM_TRIALS_PER_DIRECTION = 20
NUM_CALIBRATION_TRIALS = 3 

current_pose = None
current_orientation_euler = None

def odom_callback(msg):
    global current_pose, current_orientation_euler
    current_pose = msg.pose.pose
    orientation_q = current_pose.orientation
    orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
    current_orientation_euler = tf.transformations.euler_from_quaternion(orientation_list)

def set_robot_state(x, y, yaw_rad, model_name=ROBOT_MODEL_NAME):
    rospy.wait_for_service('/gazebo/set_model_state', timeout=5.0)
    try:
        set_state_srv = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        state_msg = ModelState()
        state_msg.model_name = model_name
        state_msg.pose.position.x = x
        state_msg.pose.position.y = y
        state_msg.pose.position.z = 0
        orientation_q = tf.transformations.quaternion_from_euler(0, 0, yaw_rad)
        state_msg.pose.orientation = Quaternion(*orientation_q)
        state_msg.twist.linear = Point(0,0,0)
        state_msg.twist.angular = Point(0,0,0)
        
        resp = set_state_srv(state_msg)
        if not resp.success:
            rospy.logwarn(f"SetModelState for {model_name} failed: {resp.status_message}")
        # else:
        #     rospy.loginfo(f"SetModelState for {model_name} to ({x:.3f},{y:.3f},{math.degrees(yaw_rad):.2f}deg) successful.")
        rospy.sleep(0.5) # Allow Gazebo and odometry to update
        return resp.success
    except rospy.ServiceException as e:
        rospy.logerr(f"Service /gazebo/set_model_state call failed: {e}")
        return False
    except rospy.ROSException as e:
        rospy.logerr(f"Service /gazebo/set_model_state not available: {e}")
        return False

def get_robot_pose():
    global current_pose, current_orientation_euler
    rospy.sleep(0.1) # Brief sleep to ensure callback has a chance to update
    if current_pose is None or current_orientation_euler is None:
        rospy.logwarn("Pose not yet available from odom_callback. Waiting...")
        for _ in range(5): # Try a few times
            rospy.sleep(0.3)
            if current_pose is not None and current_orientation_euler is not None:
                break
        if current_pose is None or current_orientation_euler is None:
            rospy.logerr("Failed to get pose from odom_callback after retries.")
            return None, None, None
    return current_pose.position.x, current_pose.position.y, current_orientation_euler[2]

def perform_rotation_for_duration(commanded_angular_vel_z, duration_sec, cmd_vel_pub):
    """
    Publishes a rotation command for a specific duration.
    Returns the actual duration the command was published.
    """
    twist_msg = Twist()
    twist_msg.angular.z = commanded_angular_vel_z
    
    start_time = rospy.Time.now()
    rate = rospy.Rate(100) # Publish at 100 Hz
    
    while not rospy.is_shutdown() and (rospy.Time.now() - start_time).to_sec() < duration_sec:
        cmd_vel_pub.publish(twist_msg)
        try:
            rate.sleep()
        except rospy.ROSInterruptException:
            rospy.logwarn("ROS interrupt during rotation sleep.")
            break 
    
    actual_published_duration = (rospy.Time.now() - start_time).to_sec()

    # Stop the robot
    twist_msg.angular.z = 0
    cmd_vel_pub.publish(twist_msg)
    rospy.sleep(0.3) # Allow odometry to update and robot to physically settle
    return actual_published_duration

def normalize_angle(angle_rad):
    while angle_rad > math.pi: angle_rad -= 2 * math.pi
    while angle_rad < -math.pi: angle_rad += 2 * math.pi
    return angle_rad

def calculate_rotation_parameters(initial_pose_tuple, final_pose_tuple, actual_delta_t_sec):
    x_i, y_i, theta_i_rad = initial_pose_tuple
    x_f, y_f, theta_f_rad = final_pose_tuple

    mu_numerator = (x_i - x_f) * math.cos(theta_i_rad) + (y_i - y_f) * math.sin(theta_i_rad)
    mu_denominator = 2 * ((y_i - y_f) * math.cos(theta_i_rad) - (x_i - x_f) * math.sin(theta_i_rad))

    if abs(mu_denominator) < 1e-9:
        mu = float('inf') if abs(mu_numerator) > 1e-9 else 0 
    else:
        mu = mu_numerator / mu_denominator

    x_star = (x_i + x_f) / 2.0 + mu * (y_i - y_f)
    y_star = (y_i + y_f) / 2.0 + mu * (x_f - x_i)
    r_star = math.sqrt((x_i - x_star)**2 + (y_i - y_star)**2)

    angle_initial_to_center = math.atan2(y_i - y_star, x_i - x_star)
    angle_final_to_center = math.atan2(y_f - y_star, x_f - x_star)
    delta_theta_actual_rad = normalize_angle(angle_final_to_center - angle_initial_to_center)

    if actual_delta_t_sec < 1e-6: # Avoid division by zero if duration is effectively zero
        omega_hat_rad_s = float('inf') if abs(delta_theta_actual_rad) > 1e-6 else 0
        gamma_hat_rad_s = float('inf') if abs(normalize_angle(theta_f_rad - theta_i_rad)) > 1e-6 else -omega_hat_rad_s
    else:
        omega_hat_rad_s = delta_theta_actual_rad / actual_delta_t_sec
        gamma_hat_rad_s = normalize_angle(theta_f_rad - theta_i_rad) / actual_delta_t_sec - omega_hat_rad_s
    
    v_hat_m_s = r_star * omega_hat_rad_s
    return mu, r_star, delta_theta_actual_rad, omega_hat_rad_s, gamma_hat_rad_s, v_hat_m_s

def plot_distribution(data_series, title_prefix, unit, filename_prefix, output_dir="."):
    if data_series.empty or data_series.isnull().all():
        rospy.logwarn(f"No valid data to plot for {title_prefix}. Skipping plot.")
        return np.nan, np.nan
    data = data_series.dropna().to_numpy()
    if data.size == 0:
        rospy.logwarn(f"No valid (non-NaN) data to plot for {title_prefix} after dropna. Skipping plot.")
        return np.nan, np.nan
    plt.figure(figsize=(12, 7))
    plt.hist(data, bins=10, density=True, alpha=0.7, color='skyblue', edgecolor='black', label='Histogram')
    data_finite = data[np.isfinite(data)]
    if data_finite.size < 2 : 
        mu_fit, std_fit = np.nan, np.nan
    else:
        mu_fit, std_fit = norm.fit(data_finite)
    if not (np.isnan(mu_fit) or np.isnan(std_fit) or std_fit <= 0):
        xmin, xmax = plt.xlim()
        x_plot = np.linspace(xmin, xmax, 200)
        p = norm.pdf(x_plot, mu_fit, std_fit)
        plt.plot(x_plot, p, 'r-', linewidth=2.5, label=f'Gaussian Fit (μ={mu_fit:.3g}, σ={std_fit:.3g})')
    mean_val = np.mean(data_finite) if data_finite.size > 0 else np.nan
    variance_val = np.var(data_finite) if data_finite.size > 0 else np.nan
    plt.title(f'{title_prefix}\nMean={mean_val:.4g} {unit}, Variance={variance_val:.4g} ({unit}²)', fontsize=14)
    plt.xlabel(f'Value ({unit})', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plot_path = os.path.join(output_dir, f"{filename_prefix}_distribution.png")
    plt.savefig(plot_path)
    plt.close()
    rospy.loginfo(f"Saved plot: {plot_path}")
    return mean_val, variance_val

def calibrate_rotation_speed(cmd_vel_pub, target_rad_s, num_cal_trials=NUM_CALIBRATION_TRIALS):
    """
    Performs calibration runs to find a correction factor for the commanded angular speed.
    """
    rospy.loginfo(f"--- Starting Speed Calibration ({num_cal_trials} trials) ---")
    rospy.loginfo(f"Targeting an actual speed of {target_rad_s:.3f} rad/s.")
    
    achieved_speeds_rad_s = []
    calibration_rotation_deg = 90.0
    calibration_angle_rad = math.radians(calibration_rotation_deg)
    
    # Initial guess for commanded speed: the target speed itself
    # We will try to find a commanded_speed_for_calibration such that the robot *actually* rotates at target_rad_s
    # For the first calibration, we command target_rad_s and see what we get.
    commanded_speed_for_calibration = target_rad_s 
    
    for i in range(num_cal_trials):
        rospy.loginfo(f"Calibration Trial {i+1}/{num_cal_trials}")
        if not set_robot_state(0, 0, 0):
            rospy.logerr("Calibration: Failed to reset robot. Skipping trial.")
            continue

        initial_x, initial_y, initial_yaw_rad = get_robot_pose()
        if initial_x is None:
            rospy.logerr("Calibration: Failed to get initial pose. Skipping trial.")
            continue

        # Determine duration based on the *commanded* speed for this calibration trial
        # and the fixed calibration angle.
        # The commanded_speed_for_calibration is what we send to Gazebo.
        if abs(commanded_speed_for_calibration) < 1e-6:
            rospy.logwarn("Calibration: Commanded speed is near zero. Cannot calculate duration.")
            continue
            
        duration_commanded = abs(calibration_angle_rad / commanded_speed_for_calibration)
        
        rospy.loginfo(f"Calibration: Commanding {commanded_speed_for_calibration:.3f} rad/s for {duration_commanded:.3f} s (target {calibration_rotation_deg} deg)")

        actual_published_duration = perform_rotation_for_duration(
            commanded_speed_for_calibration if calibration_angle_rad > 0 else -commanded_speed_for_calibration, 
            duration_commanded, 
            cmd_vel_pub
        )

        final_x, final_y, final_yaw_rad = get_robot_pose()
        if final_x is None:
            rospy.logerr("Calibration: Failed to get final pose. Skipping trial.")
            continue

        actual_rotation_rad = normalize_angle(final_yaw_rad - initial_yaw_rad)
        
        if actual_published_duration < 1e-3: # Avoid division by zero if something went wrong with timing
            actual_achieved_speed_rad_s = 0
            rospy.logwarn("Calibration: Actual published duration very small. Achieved speed set to 0.")
        else:
            actual_achieved_speed_rad_s = actual_rotation_rad / actual_published_duration
        
        achieved_speeds_rad_s.append(abs(actual_achieved_speed_rad_s)) # Store absolute speed
        rospy.loginfo(f"Calibration Trial {i+1}: Commanded: {commanded_speed_for_calibration:.3f} rad/s, "
                      f"Actual Rotation: {math.degrees(actual_rotation_rad):.2f} deg in {actual_published_duration:.3f} s, "
                      f"Achieved Speed: {actual_achieved_speed_rad_s:.3f} rad/s")

    if not achieved_speeds_rad_s:
        rospy.logwarn("Calibration failed: No speed data collected. Using correction factor 1.0.")
        return 1.0

    avg_achieved_speed = np.mean([s for s in achieved_speeds_rad_s if s > 1e-3]) # Filter out zeros or tiny speeds

    if avg_achieved_speed < 1e-3: # If average is still near zero
        rospy.logwarn(f"Calibration: Average achieved speed ({avg_achieved_speed:.4f} rad/s) is too low. "
                      "This indicates a problem with robot movement or very high scaling by Gazebo. "
                      "Using correction factor 1.0. Manual adjustment may be needed.")
        return 1.0 # Default to no correction if calibration is problematic

    correction_factor = target_rad_s / avg_achieved_speed
    
    rospy.loginfo(f"--- Calibration Complete ---")
    rospy.loginfo(f"Average achieved speed during calibration (when commanding {commanded_speed_for_calibration:.3f} rad/s): {avg_achieved_speed:.4f} rad/s")
    rospy.loginfo(f"To achieve desired {target_rad_s:.3f} rad/s, calculated command correction_factor: {correction_factor:.4f}")
    if correction_factor > 5.0 or correction_factor < 0.2:
        rospy.logwarn("Correction factor is very large or small. Double-check calibration results and robot behavior.")
    return correction_factor


def main():
    rospy.init_node('rotation_model_experiment_calibrated')

    cmd_vel_pub = rospy.Publisher(CMD_VEL_TOPIC, Twist, queue_size=10)
    rospy.Subscriber(ODOM_TOPIC, Odometry, odom_callback, queue_size=10)

    rospy.loginfo("Waiting for odometry data...")
    while current_pose is None and not rospy.is_shutdown(): rospy.sleep(0.1)
    if rospy.is_shutdown(): return
    rospy.loginfo("Odometry received.")

    # --- Perform Calibration ---
    speed_correction_factor = calibrate_rotation_speed(cmd_vel_pub, TARGET_ANGULAR_VELOCITY_RAD_S, NUM_CALIBRATION_TRIALS)
    
    # This is the angular speed we will command to ROS/Gazebo
    # to try and achieve TARGET_ANGULAR_VELOCITY_RAD_S actually.
    commanded_angular_velocity_for_experiment_rad_s = TARGET_ANGULAR_VELOCITY_RAD_S * speed_correction_factor
    rospy.loginfo(f"--- Starting Main Experiment ---")
    rospy.loginfo(f"Target actual speed: {TARGET_ANGULAR_VELOCITY_RAD_S:.3f} rad/s.")
    rospy.loginfo(f"Using corrected command speed: {commanded_angular_velocity_for_experiment_rad_s:.3f} rad/s (factor: {speed_correction_factor:.3f}).")


    results_list = []
    target_rotation_rad = math.radians(TARGET_ROTATION_DEG)

    if abs(commanded_angular_velocity_for_experiment_rad_s) < 1e-6 :
        rospy.logerr("Corrected command angular velocity is near zero. Cannot proceed with experiment.")
        return
        
    experiment_duration_sec = abs(target_rotation_rad / TARGET_ANGULAR_VELOCITY_RAD_S)
    rospy.loginfo(f"Calculated duration for {TARGET_ROTATION_DEG} deg rotations: {experiment_duration_sec:.3f} s")


    for direction_label, rotation_sign in [("CW", -1), ("CCW", 1)]:
        rospy.loginfo(f"\n--- Starting {NUM_TRIALS_PER_DIRECTION} {direction_label} Rotations ({rotation_sign*TARGET_ROTATION_DEG:.1f} deg) ---")
        for i in range(NUM_TRIALS_PER_DIRECTION):
            if rospy.is_shutdown(): break
            rospy.loginfo(f"\n{direction_label} Trial {i+1}/{NUM_TRIALS_PER_DIRECTION}")

            # Reset robot to (0,0,0) before EACH trial
            if not set_robot_state(0, 0, 0):
                rospy.logerr(f"{direction_label} Trial {i+1}: Failed to reset robot. Skipping trial.")
                continue
            
            # Get initial pose (should be very close to 0,0,0)
            initial_x, initial_y, initial_yaw_rad = get_robot_pose()
            if initial_x is None:
                rospy.logerr(f"{direction_label} Trial {i+1}: Failed to get initial pose. Skipping.")
                continue
            trial_initial_pose_tuple = (initial_x, initial_y, initial_yaw_rad)
            # Log it to confirm reset
            rospy.loginfo(f"Initial pose for trial: x={initial_x:.4f}, y={initial_y:.4f}, yaw={math.degrees(initial_yaw_rad):.2f} deg")

            # Perform rotation using the corrected angular velocity and calculated duration
            actual_published_duration = perform_rotation_for_duration(
                rotation_sign * commanded_angular_velocity_for_experiment_rad_s, 
                experiment_duration_sec, 
                cmd_vel_pub
            )
            
            final_x, final_y, final_yaw_rad = get_robot_pose()
            if final_x is None:
                rospy.logerr(f"{direction_label} Trial {i+1}: Failed to get final pose. Skipping.")
                continue
            trial_final_pose_tuple = (final_x, final_y, final_yaw_rad)
            
            # Log simple rotation
            simple_actual_rotation_deg = math.degrees(normalize_angle(final_yaw_rad - initial_yaw_rad))
            rospy.loginfo(f"Target: {rotation_sign*TARGET_ROTATION_DEG:.1f} deg. Simple odom rotation: {simple_actual_rotation_deg:.2f} deg in {actual_published_duration:.3f}s.")

            mu, r_star, delta_theta_act_rad, omega_hat_rs, gamma_hat_rs, v_hat_ms = \
                calculate_rotation_parameters(trial_initial_pose_tuple, trial_final_pose_tuple, experiment_duration_sec) # Using experiment_duration_sec
            
            results_list.append({
                'trial': i + 1, 'direction': direction_label,
                'x_initial_m': trial_initial_pose_tuple[0], 'y_initial_m': trial_initial_pose_tuple[1], 'theta_initial_rad': trial_initial_pose_tuple[2],
                'x_final_m': trial_final_pose_tuple[0], 'y_final_m': trial_final_pose_tuple[1], 'theta_final_rad': trial_final_pose_tuple[2],
                'commanded_duration_s': experiment_duration_sec, # This is the delta_t for the formulas
                'actual_cmd_publish_duration_s': actual_published_duration, # For debugging timing
                'mu_delta': mu, 'r_star_m': r_star, 'delta_theta_actual_rad': delta_theta_act_rad,
                'omega_hat_rad_s': omega_hat_rs, 'gamma_hat_rad_s': gamma_hat_rs, 'v_hat_m_s': v_hat_ms
            })
            rospy.loginfo(f"{direction_label} Trial {i+1} Data: μ={mu:.4f}, r*={r_star:.4f}m, Δθ_act={math.degrees(delta_theta_act_rad):.2f}°, "
                          f"ῶ={omega_hat_rs:.4f}rad/s, ŷ={gamma_hat_rs:.4f}rad/s, ύ={v_hat_ms:.4f}m/s")

    if not results_list:
        rospy.logerr("No data collected. Exiting.")
        return

    df = pd.DataFrame(results_list)
    output_csv_path = "rotation_experiment.csv"
    df.to_csv(output_csv_path, index=False, float_format='%.6f')
    rospy.loginfo(f"All data saved to {output_csv_path}")

    output_dir_cw = "plots_rotation_cw"
    output_dir_ccw = "plots_rotation_ccw"
    os.makedirs(output_dir_cw, exist_ok=True)
    os.makedirs(output_dir_ccw, exist_ok=True)

    analysis_params = { # ... same as before ...
        'mu_delta': {'unit': '', 'name': 'μ (delta)'},
        'r_star_m': {'unit': 'm', 'name': 'r* (Effective Radius)'},
        'omega_hat_rad_s': {'unit': 'rad/s', 'name': 'ω̂ (Omega_hat / Effective Ang. Vel.)'},
        'gamma_hat_rad_s': {'unit': 'rad/s', 'name': 'ŷ (Gamma_hat / Direct Odom Ang. Vel.)'},
        'v_hat_m_s': {'unit': 'm/s', 'name': 'v̂ (V_hat / Effective Tangential Speed)'}
    }
    df['omega_hat_deg_s'] = np.degrees(df['omega_hat_rad_s'])
    df['gamma_hat_deg_s'] = np.degrees(df['gamma_hat_rad_s'])
    analysis_params_deg = { # ... same as before ...
        'omega_hat_deg_s': {'unit': 'deg/s', 'name': 'ῶ (Omega_hat / Effective Ang. Vel.)'},
        'gamma_hat_deg_s': {'unit': 'deg/s', 'name': 'ŷ (Gamma_hat / Direct Odom Ang. Vel.)'}
    }

    for dir_label, df_sub, out_dir in [("CW", df[df['direction'] == 'CW'], output_dir_cw), 
                                       ("CCW", df[df['direction'] == 'CCW'], output_dir_ccw)]:
        if df_sub.empty: continue
        rospy.loginfo(f"\n--- {dir_label} Rotation Analysis (Calibrated, Reset Each Trial) ---")
        for p_key, props in analysis_params.items():
            m, v = plot_distribution(df_sub[p_key], f"{props['name']} - {dir_label}", props['unit'], 
                                     f"{p_key.replace('_rad_s','').replace('_m_s','').replace('_m','')}_{dir_label.lower()[:3]}", out_dir)
            rospy.loginfo(f"{dir_label} {props['name']} ({props['unit']}): Mean={m:.4g}, Variance={v:.4g}")
        for p_key, props in analysis_params_deg.items():
            m, v = plot_distribution(df_sub[p_key], f"{props['name']} - {dir_label}", props['unit'], 
                                     f"{p_key.replace('_deg_s','')}_{dir_label.lower()[:3]}_deg", out_dir)
            rospy.loginfo(f"{dir_label} {props['name']} ({props['unit']}): Mean={m:.4g}, Variance={v:.4g}")
    rospy.loginfo("Experiment finished.")

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException: rospy.loginfo("ROS interrupt. Shutting down.")
    except Exception as e:
        rospy.logerr(f"Unhandled exception in main: {e}")
        import traceback
        traceback.print_exc()