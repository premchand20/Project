#importing Libraries
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageFilter, ImageOps
import numpy as np
import cv2
from skimage.morphology import closing, square
import os

class MainApplication:
    def __init__(self, root):
        self.root = root
        self.root.title("Object Detection and Classification")

        # Create buttons
        self.object_detection_button = tk.Button(root, text="Object Detection", command=self.object_detection)
        self.object_detection_button.pack(pady=10)

        self.object_classification_button = tk.Button(root, text="Object Classification", command=self.object_classification)
        self.object_classification_button.pack(pady=10)

        # Create navigation buttons
        self.home_button = tk.Button(root, text="Home", command=self.go_home)
        self.home_button.pack(side=tk.LEFT, padx=10)

        self.back_button = tk.Button(root, text="Back", command=self.go_back)
        self.back_button.pack(side=tk.LEFT, padx=10)

        self.next_button = tk.Button(root, text="Next", command=self.go_next)
        self.next_button.pack(side=tk.LEFT, padx=10)

        # Variables for image processing
        self.image_paths = []
        self.current_image_index = 0
        self.processed_images = []
        self.processing_steps = [
            ("Resized Image", "Resizing the image to 512x512"),
            ("Noise Removal", "Applying median filter for Noise Removal"),
            ("Image Enhancement", "Applying unsharp masking for Image Enhancement"),
            ("Tree Canopy Segmentation", "Performing Tree Canopy Segmentation"),
            ("Edge Detection", "Applying Canny Edge Detection"),
            ("Display Objects", "Applying Morphological Closing Operation and Displaying Objects"),
            # Add more processing steps as needed
        ]
        self.current_processing_step = tk.StringVar()
        self.processing_step_label = tk.Label(root, textvariable=self.current_processing_step)
        self.processing_step_label.pack()

        # Create a frame to contain the image label and processing step label
        self.image_frame = tk.Frame(root)
        self.image_frame.pack()

    def object_detection(self):
        # Open a file dialog to choose image files
        file_paths = filedialog.askopenfilenames(title="Select Image Files", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])

        if file_paths:
            self.image_paths = list(file_paths)
            self.current_image_index = 0
            self.loaded_image = cv2.imread(self.image_paths[0])
            self.process_images()

    def process_images(self):
        # Reset processed images
        self.processed_images = []

        for image_path in self.image_paths:
            # Open image
            original_image = Image.open(image_path)

            # Resize image to 512x512
            resized_image = original_image.resize((512, 512))
            self.processed_images.append(resized_image)

            # Apply noise removal using median filter
            noise_removal_image = self.apply_median_filter(resized_image)
            self.processed_images.append(noise_removal_image)

            # Apply image enhancement using unsharp masking
            enhanced_image = self.apply_unsharp_masking(noise_removal_image)
            self.processed_images.append(enhanced_image)

            # Perform tree canopy segmentation
            segmented_image = self.perform_tree_canopy_segmentation(enhanced_image)
            self.processed_images.append(segmented_image)

            # Perform edge detection using Canny Edge Detection
            edge_image = self.perform_edge_detection(resized_image)
            self.processed_images.append(edge_image)

            # Perform morphological operations
            morphological_image = self.apply_morphological_operations(enhanced_image)
            self.processed_images.append(morphological_image)

        # Display the first processed image
        self.display_image()

    def apply_median_filter(self, image):
        # Apply median filter for noise removal
        return image.filter(ImageFilter.MedianFilter(size=3))

    def apply_unsharp_masking(self, image):
        # Apply unsharp masking for image enhancement
        return image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    
    def perform_tree_canopy_segmentation(self, image):
        # Convert the enhanced image to grayscale
        grayscale_image = ImageOps.grayscale(image)

        # Convert the image to a NumPy array for easier manipulation
        image_array = np.array(grayscale_image)

        # Apply a simple threshold to identify green areas (adjust threshold as needed)
        threshold = 128
        binary_image = (image_array > threshold).astype(np.uint8) * 255

        # Convert the binary image back to a PIL Image
        segmented_image = Image.fromarray(binary_image, mode='L')

        # Apply a color map for better visualization (replace with your color map)
        segmented_image = segmented_image.convert('P', palette=Image.ADAPTIVE, colors=256)

        # Convert the indexed image to a colored image
        colored_image = segmented_image.convert('RGB')

        # Define a color map (replace with your desired colors)
        colormap = {
            0: (255, 255, 0),  # Background (yellow)
            255: (0, 255, 0)    # Tree canopies (green)
        }

        # Apply the color map to the image
        colored_image = colored_image.convert('P', palette=Image.ADAPTIVE, colors=256)
        colored_image.putpalette([col for rgb in colormap.values() for col in rgb])

        return colored_image

    def perform_edge_detection(self,image):
        # Convert the PIL Image to a NumPy array
        numpy_image = np.array(image)

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2GRAY)

        # Apply Canny Edge Detection
        edges = cv2.Canny(gray_image, 50, 150)

        # Convert the NumPy array back to a PIL Image
        edge_image = Image.fromarray(edges)

        return edge_image

    def apply_morphological_operations(self, image):
        # Convert the enhanced image to grayscale
        grayscale_image = ImageOps.grayscale(image)

        # Convert the image to a NumPy array for easier manipulation
        image_array = np.array(grayscale_image)

        # Apply a simple threshold to identify green areas (adjust threshold as needed)
        threshold = 128
        binary_image = (image_array > threshold).astype(np.uint8) * 255

        # Convert the binary image back to a PIL Image
        morphological_image = Image.fromarray(binary_image, mode='L')

        # Apply a color map for better visualization (replace with your color map)
        morphological_image = morphological_image.convert('P', palette=Image.ADAPTIVE, colors=256)

        # Convert the indexed image to a colored image
        colored_image = morphological_image.convert('RGB')

        # Define a color map (replace with your desired colors)
        colormap = {
            0: (255, 255, 255),  # Background (Black)
            255: (0, 0, 0)    #Objects (White)
        }

        # Apply the color map to the image
        colored_image = colored_image.convert('P', palette=Image.ADAPTIVE, colors=256)
        colored_image.putpalette([col for rgb in colormap.values() for col in rgb])

        return colored_image

    def display_image(self):
        # Clear previous image
        for widget in self.image_frame.winfo_children():
            widget.destroy()

        # Display the current processed image
        current_image = self.processed_images[self.current_image_index]
        photo = ImageTk.PhotoImage(current_image)
        label = tk.Label(self.image_frame, image=photo)
        label.image = photo
        label.pack()

        # Display the processing step message at the bottom
        processing_step_message = self.processing_steps[self.current_image_index][1]
        processing_step_label = tk.Label(self.image_frame, text=processing_step_message)
        processing_step_label.pack(side=tk.BOTTOM)

    def go_home(self):
        self.image_paths = []
        self.processed_images = []
        self.current_image_index = 0
        for widget in self.image_frame.winfo_children():
            widget.destroy()

    def go_back(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.display_image()

    def go_next(self):
        if self.current_image_index < len(self.processed_images) - 1:
            self.current_image_index += 1
            self.current_processing_step.set(self.processing_steps[self.current_image_index][0])
            self.display_image()

    def object_classification(self):
        # Open a file dialog to choose an input image file
        input_image_path = filedialog.askopenfilename(title="Select Input Image", filetypes=[("Image files", "*.tif;*.jpg;*.jpeg")])

        if input_image_path:
            # Read the input image
            input_image = Image.open(input_image_path)

            # Get the numeric identifier from the input image name
            input_image_name = os.path.splitext(os.path.basename(input_image_path))[0]
            input_numeric_identifier = ''.join(filter(str.isdigit, input_image_name))

            # Perform object classification logic (replace this with your actual implementation)
            output_image = self.perform_object_classification(input_numeric_identifier)

            # Display the input and output images
            self.display_input_output_images(input_image, output_image)

    def perform_object_classification(self, numeric_identifier):
        # Replace this with your actual object classification logic
        # For demonstration purposes, loading a dummy output image from the masks folder
        masks_folder_path = "F:\Major\classification imgs\masks"
        output_image_path = os.path.join(masks_folder_path, f"{numeric_identifier}.tif")

        if os.path.exists(output_image_path):
            output_image = Image.open(output_image_path)
            return output_image
        else:
            print(f"Output image not found for identifier {numeric_identifier}")
            return None

    def display_input_output_images(self, input_image, output_image):
        # Clear previous images
        for widget in self.image_frame.winfo_children():
            widget.destroy()

         # Display the input image
        input_photo = ImageTk.PhotoImage(input_image)
        input_label = tk.Label(self.image_frame, image=input_photo, text="Input Image", compound=tk.BOTTOM)
        input_label.image = input_photo
        input_label.pack(side=tk.LEFT, padx=10)

        # Display the output image
        if output_image:
            output_photo = ImageTk.PhotoImage(output_image)
            output_label = tk.Label(self.image_frame, image=output_photo, text="Classified Image", compound=tk.BOTTOM)
            output_label.image = output_photo
            output_label.pack(side=tk.LEFT, padx=10)
        
        
if __name__ == "__main__":
    root = tk.Tk()
    app = MainApplication(root)
    root.mainloop()
