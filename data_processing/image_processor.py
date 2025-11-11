import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class ImageProcessor:
    """Process waveform images from datasheets, screenshots, etc."""
    
    def __init__(self):
        self.supported_formats = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    
    def extract_waveform(self, image_path, method='projection'):
        """Extract waveform data from image"""
        image = self._load_and_preprocess(image_path)
        
        if method == 'projection':
            return self._extract_by_projection(image)
        elif method == 'contour':
            return self._extract_by_contour(image)
        else:
            raise ValueError(f"Unknown extraction method: {method}")
    
    def _load_and_preprocess(self, image_path):
        """Load and preprocess image for waveform extraction"""
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        return enhanced
    
    def _extract_by_projection(self, image):
        """Extract waveform using horizontal projection"""
        height, width = image.shape
        
        # Binarize image
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Invert if background is dark
        if np.mean(binary) < 127:
            binary = cv2.bitwise_not(binary)
        
        # Extract waveform points column by column
        time_points = []
        amplitude_points = []
        
        for col in range(width):
            column = binary[:, col]
            white_pixels = np.where(column > 127)[0]
            
            if len(white_pixels) > 0:
                # Use centroid of white pixels in this column
                centroid = np.mean(white_pixels)
                time_points.append(col / width)  # Normalized time
                amplitude_points.append(1.0 - (centroid / height))  # Normalized amplitude
        
        return np.array(time_points), np.array(amplitude_points)
    
    def _extract_by_contour(self, image):
        """Extract waveform using contour detection"""
        height, width = image.shape
        
        # Binarize and find contours
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        if np.mean(binary) < 127:
            binary = cv2.bitwise_not(binary)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return np.array([]), np.array([])
        
        # Find the largest contour (assumed to be the waveform)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Extract points from contour
        points = largest_contour.reshape(-1, 2)
        time_points = points[:, 0] / width  # Normalized X
        amplitude_points = 1.0 - (points[:, 1] / height)  # Normalized Y
        
        # Sort by time
        sort_idx = np.argsort(time_points)
        time_points = time_points[sort_idx]
        amplitude_points = amplitude_points[sort_idx]
        
        return time_points, amplitude_points
    
    def detect_grid(self, image):
        """Detect and remove grid lines from oscilloscope images"""
        # Use Hough line transform to detect grid lines
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is not None:
            # Create mask for grid lines
            mask = np.ones_like(image) * 255
            
            for rho, theta in lines[:, 0]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                
                cv2.line(mask, (x1, y1), (x2, y2), 0, 2)
            
            # Remove grid lines
            result = cv2.bitwise_and(image, mask)
            return result
        
        return image
    
    def calibrate_axes(self, image, known_points):
        """Calibrate image axes using known reference points"""
        # This would use known voltage/time references to calibrate the axes
        # For now, return identity calibration
        return {
            'time_scale': 1.0,
            'amplitude_scale': 1.0,
            'time_offset': 0.0,
            'amplitude_offset': 0.0
        }