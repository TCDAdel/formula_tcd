import numpy as np

class Driver:
    CAR_WIDTH = 0.06
    DIFFERENCE_THRESHOLD = 0.6
    SPEED = 0.9
    SAFETY_PERCENTAGE = 200
    MAX_DISTANCE = 50.0  # Maximum LiDAR detection distance in meters

    def preprocess_lidar(self, ranges):
        ranges = np.array(ranges)
        eighth = len(ranges) // 8
        # Focus on the forward 180 degrees, removing data directly behind the car
        ranges = ranges[eighth:-eighth]
        # Capping the distance at MAX_DISTANCE
        ranges = np.clip(ranges, None, self.MAX_DISTANCE)
        # smoothing technique
        smoothed_ranges = np.convolve(ranges, np.ones(5)/5, mode='same')
        return smoothed_ranges


    def get_differences(self, ranges):
        """Vectorized computation of absolute differences between adjacent LiDAR points."""
        differences = np.abs(np.diff(ranges, prepend=ranges[0]))
        return differences

    def get_disparities(self, differences, threshold):
        """Identifies indices of significant disparities using vectorized operations."""
        disparities = np.where(differences > threshold)[0]
        return disparities

    def get_num_points_to_cover(self, dist, width):
        """Calculates the number of LiDAR points covering a given width at a distance."""
        angle = 2 * np.arctan(width / (2 * dist))
        num_points = int(np.ceil(angle / self.radians_per_point))
        return num_points

    def cover_points(self, num_points, start_idx, cover_right, ranges):
        """Optimized coverage of points using slice assignments."""
        new_dist = ranges[start_idx]
        if cover_right:
            end_idx = min(start_idx + 1 + num_points, len(ranges))
            ranges[start_idx + 1:end_idx] = np.minimum(ranges[start_idx + 1:end_idx], new_dist)
        else:
            start = max(start_idx - num_points, 0)
            ranges[start:start_idx] = np.minimum(ranges[start:start_idx], new_dist)
        return ranges

    def extend_disparities(self, disparities, ranges, car_width, extra_pct):
        """Extends disparities with optimized vectorized operations."""
        width_to_cover = (car_width / 2) * (1 + extra_pct / 100)
        for index in disparities:
            points = ranges[index - 1:index + 1]
            close_idx, far_idx = np.argmin(points) + index - 1, np.argmax(points) + index - 1
            close_dist = ranges[close_idx]
            num_points_to_cover = self.get_num_points_to_cover(close_dist, width_to_cover)
            cover_right = close_idx < far_idx
            ranges = self.cover_points(num_points_to_cover, close_idx, cover_right, ranges)
        return ranges

    def get_steering_angle(self, range_index, range_len):
        """Calculates the steering angle for a given LiDAR point."""
        lidar_angle = (range_index - (range_len / 2)) * self.radians_per_point
        steering_angle = np.clip(lidar_angle, np.radians(-90), np.radians(90))
        return steering_angle

    def process_lidar(self, ranges):
        """Processes LiDAR data to determine steering and speed adjustments."""
        self.radians_per_point = (2 * np.pi) / len(ranges)
        proc_ranges = self.preprocess_lidar(ranges)
        differences = self.get_differences(proc_ranges)
        disparities = self.get_disparities(differences, self.DIFFERENCE_THRESHOLD)
        proc_ranges = self.extend_disparities(disparities, proc_ranges, self.CAR_WIDTH, self.SAFETY_PERCENTAGE)
        steering_angle = self.get_steering_angle(proc_ranges.argmax(), len(proc_ranges))
        speed = self.SPEED * 5 * (1 - abs(steering_angle) / (np.pi))
        return speed, steering_angle
    
    
    def adjust_speed_multiplier(self, steering_angle):
        # Simple heuristic: Decrease speed as the steering angle increases
        if abs(steering_angle) < np.radians(10):
            return 5  # Straight path, increase speed
        elif abs(steering_angle) < np.radians(30):
            return 3  # Moderate curve, moderate speed
        else:
            return 1  # Sharp curve, reduce speed
        
    def adjust_safety_margin(self, speed):
        # Adjust safety margin based on speed: Higher speed requires higher margin
        if speed > 1.5:  # Assuming speed scale is 0-5
            return 325  # Increase safety margin at high speeds
        else:
            return 200  # Standard safety margin

