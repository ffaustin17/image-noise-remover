import tkinter as tk
from tkinter import font
from PIL import Image, ImageTk, ImageDraw
import math


# Helper function to read and convert an image to grayscale as a 2D list
def read_image(filepath):
    """Reads an image file and converts it to grayscale as a 2D list."""
    image = Image.open(filepath).convert("L")
    width, height = image.size
    pixels = list(image.getdata())

    # Replace list comprehension for generating 2D list
    pixel_rows = []
    for i in range(height):
        row = []
        for j in range(width):
            row.append(pixels[i * width + j])
        pixel_rows.append(row)

    return pixel_rows, width, height


# Functions for image processing
def calculate_mse(image1, image2):
    """Calculates Mean Squared Error between two images."""
    height = len(image1)
    width = len(image1[0])
    error = 0
    for i in range(height):
        for j in range(width):
            error += (image1[i][j] - image2[i][j]) ** 2
    return error / (height * width)


def calculate_snr(original, noisy):
    """Calculates Signal-to-Noise Ratio (SNR)."""
    signal_power = 0
    noise_power = 0
    height = len(original)
    width = len(original[0])
    for i in range(height):
        for j in range(width):
            signal_power += original[i][j] ** 2
            noise_power += (original[i][j] - noisy[i][j]) ** 2
    return 10 * math.log10(signal_power / noise_power)


def calculate_psnr(original, noisy):
    """Calculates Peak Signal-to-Noise Ratio (PSNR)."""
    mse = calculate_mse(original, noisy)
    if mse == 0:
        return float("inf")
    max_pixel = 255.0
    return 20 * math.log10(max_pixel / math.sqrt(mse))


def apply_average_filter(image):
    """Applies a 3x3 average filter."""
    height = len(image)
    width = len(image[0])
    result = []
    for _ in range(height):
        result.append([0] * width)

    # Replace list comprehension with explicit loops for padding
    padded_image = [[0] * (width + 2)]
    for row in image:
        padded_row = [0] + row + [0]
        padded_image.append(padded_row)
    padded_image.append([0] * (width + 2))

    for i in range(1, height + 1):
        for j in range(1, width + 1):
            neighbors = [
                padded_image[i-1][j-1], padded_image[i-1][j], padded_image[i-1][j+1],
                padded_image[i][j-1], padded_image[i][j], padded_image[i][j+1],
                padded_image[i+1][j-1], padded_image[i+1][j], padded_image[i+1][j+1]
            ]
            result[i-1][j-1] = sum(neighbors) // len(neighbors)
    return result


def apply_median_filter(image):
    """Applies a 3x3 median filter."""
    height = len(image)
    width = len(image[0])
    result = []
    for _ in range(height):
        result.append([0] * width)

    # Replace list comprehension with explicit loops for padding
    padded_image = [[0] * (width + 2)]
    for row in image:
        padded_row = [0] + row + [0]
        padded_image.append(padded_row)
    padded_image.append([0] * (width + 2))

    for i in range(1, height + 1):
        for j in range(1, width + 1):
            neighbors = [
                padded_image[i-1][j-1], padded_image[i-1][j], padded_image[i-1][j+1],
                padded_image[i][j-1], padded_image[i][j], padded_image[i][j+1],
                padded_image[i+1][j-1], padded_image[i+1][j], padded_image[i+1][j+1]
            ]
            result[i-1][j-1] = sorted(neighbors)[len(neighbors) // 2]
    return result


def detect_noise_dynamic(image):
    """
    Detects noisy pixels using a dynamic threshold based on local mean and standard deviation.
    """
    height = len(image)
    width = len(image[0])
    noise_mask = []
    for _ in range(height):
        noise_mask.append([0] * width)

    # Replace list comprehension with explicit loops for padding
    padded_image = [[0] * (width + 2)]
    for row in image:
        padded_row = [0] + row + [0]
        padded_image.append(padded_row)
    padded_image.append([0] * (width + 2))

    for i in range(1, height + 1):
        for j in range(1, width + 1):
            neighbors = [
                padded_image[i-1][j-1], padded_image[i-1][j], padded_image[i-1][j+1],
                padded_image[i][j-1], padded_image[i][j], padded_image[i][j+1],
                padded_image[i+1][j-1], padded_image[i+1][j], padded_image[i+1][j+1]
            ]
            local_mean = sum(neighbors) / len(neighbors)
            local_std_dev = (sum((x - local_mean) ** 2 for x in neighbors) / len(neighbors)) ** 0.5

            if abs(image[i-1][j-1] - local_mean) > local_std_dev:
                noise_mask[i-1][j-1] = 1

    return noise_mask

def visualize_noise(image, noise_mask):
    """Creates a visualization of noise in the image."""
    height = len(image)
    width = len(image[0])
    visual_image = Image.new("RGB", (width, height), color=(0, 255, 0))  # Green background
    draw = ImageDraw.Draw(visual_image)
    for i in range(height):
        for j in range(width):
            if noise_mask[i][j] == 1:
                draw.point((j, i), fill=(255, 0, 0))  # Red for noise
    return visual_image


def apply_median_filter_on_noise(image, noise_mask):
    """
    Applies a median filter only to the pixels identified as noisy in the noise mask.
    Args:
        image (list): 2D list of the original image's pixel values.
        noise_mask (list): 2D binary mask (1 for noise, 0 for non-noise).
    Returns:
        list: A 2D list representing the processed image.
    """
    height = len(image)
    width = len(image[0])

    # Create a copy of the image to store the result
    result = []
    for row in image:
        result_row = []
        for pixel in row:
            result_row.append(pixel)
        result.append(result_row)

    # Pad the image to handle edge cases
    padded_image = [[0] * (width + 2)]
    for row in image:
        padded_row = [0] + row + [0]
        padded_image.append(padded_row)
    padded_image.append([0] * (width + 2))

    for i in range(1, height + 1):
        for j in range(1, width + 1):
            if noise_mask[i-1][j-1] == 1:  # If the pixel is noisy
                # Extract the 3x3 neighborhood
                neighbors = [
                    padded_image[i-1][j-1], padded_image[i-1][j], padded_image[i-1][j+1],
                    padded_image[i][j-1], padded_image[i][j], padded_image[i][j+1],
                    padded_image[i+1][j-1], padded_image[i+1][j], padded_image[i+1][j+1]
                ]
                # Replace the pixel with the median of its neighborhood
                result[i-1][j-1] = sorted(neighbors)[len(neighbors) // 2]

    return result


class NoiseFilteringApp:
    def __init__(self, root, original_image_path, noisy_image_path):
        self.root = root
        self.root.title("Noise Filter")
        
        # Load images from file paths
        self.original_image, self.width, self.height = read_image(original_image_path)
        self.noisy_image, _, _ = read_image(noisy_image_path)
        self.processed_image = None  # Holds the result of the last operation
        
        # Track whether each processed image has been saved
        self.saved_images = {
            "Lena_average": False,
            "Lena_median": False,
            "Lena_median_filtered_detected_noise": False,
            "Lena_median_filtered_actual_noise": False,
        }
        self.setup_ui()

    def setup_ui(self):
        """Sets up the user interface."""
        # Frame for all images
        self.image_frame = tk.Frame(self.root)
        self.image_frame.pack(side=tk.TOP, pady=10)

        bold_font = font.Font(weight="bold")

        # Original Image
        self.original_label = tk.Label(self.image_frame, text="Original Image", font=bold_font)
        self.original_label.grid(row=0, column=0)
        self.original_image_label = tk.Label(self.image_frame)
        self.original_image_label.grid(row=1, column=0)

        # Noisy Image
        self.noisy_label = tk.Label(self.image_frame, text="Noisy Image", font=bold_font)
        self.noisy_label.grid(row=0, column=1)
        self.noisy_image_label = tk.Label(self.image_frame)
        self.noisy_image_label.grid(row=1, column=1)

        # Processed Image
        self.result_label = tk.Label(self.image_frame, text="Processed Image", font=bold_font)
        self.result_label.grid(row=0, column=2)
        self.result_image_label = tk.Label(self.image_frame)
        self.result_image_label.grid(row=1, column=2)

        # Display the original and noisy images
        self.display_image(self.original_image, self.original_image_label)
        self.display_image(self.noisy_image, self.noisy_image_label)

        # Buttons for operations
        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack(side=tk.TOP, pady=10)

        self.average_button = tk.Button(self.button_frame, text="Apply Average Filter", command=self.apply_average_filter)
        self.average_button.pack(side=tk.LEFT, padx=5)

        self.median_button = tk.Button(self.button_frame, text="Apply Median Filter", command=self.apply_median_filter)
        self.median_button.pack(side=tk.LEFT, padx=5)

        self.targeted_median_button = tk.Button(self.button_frame, text="Targeted Median Filter", command=self.apply_targeted_median_filter)
        self.targeted_median_button.pack(side=tk.LEFT, padx=5)

        self.show_noise_button = tk.Button(self.button_frame, text="Show Noise", command=self.show_noise)
        self.show_noise_button.pack(side=tk.LEFT, padx=5)

        self.show_actual_noise_button = tk.Button(self.button_frame, text="Show Actual Noise", command=self.show_actual_noise)
        self.show_actual_noise_button.pack(side=tk.LEFT, padx=5)

        self.targeted_actual_noise_button = tk.Button(self.button_frame, text="Targeted Filter on Actual Noise", command=self.targeted_filter_actual_noise)
        self.targeted_actual_noise_button.pack(side=tk.LEFT, padx=5)

        # Metrics frame (directly below images)
        self.metrics_frame = tk.Frame(self.root)
        self.metrics_frame.pack(side=tk.TOP, pady=10)

        # Metrics Labels
        self.metrics_noisy_label = tk.Label(self.metrics_frame, text="", justify=tk.LEFT, font=("Helvetica", 12))
        self.metrics_noisy_label.pack(anchor="w")
        self.metrics_processed_label = tk.Label(self.metrics_frame, text="", justify=tk.LEFT, font=("Helvetica", 12))
        self.metrics_processed_label.pack(anchor="w")

        # Initial metrics display
        self.update_metrics()

    def save_image(self, image_array, filename):
        """Saves a processed image to disk."""

        output_path = "./images/output/"

        if not self.saved_images[filename]:  # Ensure the image is saved only once
            image = Image.new("L", (self.width, self.height))
            pixels = []
            for row in image_array:
                for pixel in row:
                    pixels.append(pixel)
            image.putdata(pixels)
            image.save(f"{output_path+filename}.png")
            self.saved_images[filename] = True  # Mark as saved

    def display_image(self, image_array, label):
        """Displays an image on the given label."""
        image = Image.new("L", (self.width, self.height))
        pixels = []
        for row in image_array:
            for pixel in row:
                pixels.append(pixel)
        image.putdata(pixels)
        image.thumbnail((300, 300))
        photo = ImageTk.PhotoImage(image)
        label.config(image=photo)
        label.image = photo

    def detect_actual_noise(self):
        """Detects actual noise by comparing original and noisy images."""
        noise_mask = []
        for _ in range(self.height):
            row = [0] * self.width
            noise_mask.append(row)

        for i in range(self.height):
            for j in range(self.width):
                if self.noisy_image[i][j] != self.original_image[i][j]:
                    noise_mask[i][j] = 1
        return noise_mask

    def apply_average_filter(self):
        """Applies the average filter to the noisy image."""
        self.processed_image = apply_average_filter(self.noisy_image)
        self.display_image(self.processed_image, self.result_image_label)
        self.save_image(self.processed_image, "Lena_average")
        self.update_metrics()

    def apply_median_filter(self):
        """Applies the median filter to the noisy image."""
        self.processed_image = apply_median_filter(self.noisy_image)
        self.display_image(self.processed_image, self.result_image_label)
        self.save_image(self.processed_image, "Lena_median")
        self.update_metrics()

    def apply_targeted_median_filter(self):
        """Applies the median filter only to noisy pixels in the noisy image."""
        noise_mask = detect_noise_dynamic(self.noisy_image)  # Detect noise
        self.processed_image = apply_median_filter_on_noise(self.noisy_image, noise_mask)
        self.display_image(self.processed_image, self.result_image_label)
        self.save_image(self.processed_image, "Lena_median_filtered_detected_noise")
        self.update_metrics()

    def show_noise(self):
        """Detects and visualizes noise in the noisy image."""
        noise_mask = detect_noise_dynamic(self.noisy_image)  # Detect noise
        visual_image = visualize_noise(self.noisy_image, noise_mask)  # Create visualization
        visual_image.thumbnail((300, 300))  # Resize for display
        photo = ImageTk.PhotoImage(visual_image)
        self.result_image_label.config(image=photo)
        self.result_image_label.image = photo
        self.processed_image = None  # Mark that this is not a processed image for metrics
        self.update_metrics()

    def show_actual_noise(self):
        """Visualizes the actual noise by using the detect_actual_noise method."""
        noise_mask = self.detect_actual_noise()
        visual_image = visualize_noise(self.noisy_image, noise_mask)  # Create visualization
        visual_image.thumbnail((300, 300))  # Resize for display
        photo = ImageTk.PhotoImage(visual_image)
        self.result_image_label.config(image=photo)
        self.result_image_label.image = photo
        self.processed_image = None  # Mark that this is not a processed image for metrics
        self.update_metrics()

    def targeted_filter_actual_noise(self):
        """Applies a targeted median filter only on actual noise."""
        noise_mask = self.detect_actual_noise()
        self.processed_image = apply_median_filter_on_noise(self.noisy_image, noise_mask)
        self.display_image(self.processed_image, self.result_image_label)
        self.save_image(self.processed_image, "Lena_median_filtered_actual_noise")
        self.update_metrics()

    def update_metrics(self):
        """Updates and displays metrics based on the current state of the app."""
        # Metrics between original and noisy image
        mse_noisy = calculate_mse(self.original_image, self.noisy_image)
        psnr_noisy = calculate_psnr(self.original_image, self.noisy_image)
        snr_noisy = calculate_snr(self.original_image, self.noisy_image)

        metrics_noisy = (
            f"Original vs Noisy\n"
            f"MSE: {mse_noisy:.2f}\n"
            f"PSNR: {psnr_noisy:.2f} dB\n"
            f"SNR: {snr_noisy:.2f} dB"
        )
        self.metrics_noisy_label.config(text=metrics_noisy, font=font.Font(size=12, weight="bold"))

        # If a processed image exists and it's not the noise visualization, add its metrics
        if self.processed_image is not None:
            mse_processed = calculate_mse(self.original_image, self.processed_image)
            psnr_processed = calculate_psnr(self.original_image, self.processed_image)
            snr_processed = calculate_snr(self.original_image, self.processed_image)

            metrics_processed = (
                f"Original vs Processed\n"
                f"MSE: {mse_processed:.2f}\n"
                f"PSNR: {psnr_processed:.2f} dB\n"
                f"SNR: {snr_processed:.2f} dB"
            )
            self.metrics_processed_label.config(text=metrics_processed)
        else:
            self.metrics_processed_label.config(text="")  # Clear processed metrics if no processed image


# Initialize and run the app
if __name__ == "__main__":
    # File paths for the original and noisy images
    original_image_path = "./images/input/Lena.jpg"  # Replace with the actual path
    noisy_image_path = "./images/input/Lena_noise.jpg"  # Replace with the actual path

    root = tk.Tk()
    app = NoiseFilteringApp(root, original_image_path, noisy_image_path)
    root.mainloop()

