
import cv2
import numpy as np
import re

class DatasheetParser:
    """Parse information from datasheet images"""
    
    def __init__(self):
        self.parameter_patterns = {
            'mosfet': {
                'Rds_on': r'Rds\(on\)\s*[=:]?\s*([0-9.]+)\s*[mΩΩ]',
                'Vth': r'Vth\s*[=:]?\s*([0-9.]+)\s*V',
                'Ciss': r'Ciss\s*[=:]?\s*([0-9.]+)\s*[pP]F',
            },
            'opamp': {
                'GBW': r'GBW\s*[=:]?\s*([0-9.]+)\s*[mMkK]?Hz',
                'slew_rate': r'slew\s*rate\s*[=:]?\s*([0-9.]+)\s*V/μs',
            }
        }
    
    def extract_parameters_from_text(self, text, component_type):
        """Extract parameters from text using regex patterns"""
        parameters = {}
        
        if component_type not in self.parameter_patterns:
            return parameters
        
        patterns = self.parameter_patterns[component_type]
        
        for param_name, pattern in patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    parameters[param_name] = float(matches[0])
                except ValueError:
                    continue
        
        return parameters
    
    def extract_text_from_image(self, image_path):
        """Extract text from datasheet image using OCR"""
        # This would use Tesseract OCR in practice
        # For now, return empty string
        try:
            import pytesseract
            image = cv2.imread(image_path)
            text = pytesseract.image_to_string(image)
            return text
        except ImportError:
            return ""
    
    def parse_datasheet(self, image_path, component_type):
        """Parse datasheet image for parameters and curves"""
        results = {
            'parameters': {},
            'curves_found': [],
            'tables_found': []
        }
        
        # Extract text
        text = self.extract_text_from_image(image_path)
        if text:
            results['parameters'] = self.extract_parameters_from_text(text, component_type)
        
        # Detect graphs and tables (simplified)
        image = cv2.imread(image_path)
        if image is not None:
            results['curves_found'] = self._detect_curves(image)
            results['tables_found'] = self._detect_tables(image)
        
        return results
    
    def _detect_curves(self, image):
        """Detect potential waveform curves in image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours that might be curves
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        curves = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 100 < area < 10000:  # Reasonable size for curves
                curves.append(contour)
        
        return len(curves)
    
    def _detect_tables(self, image):
        """Detect potential tables in image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use Hough lines to detect table structure
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is not None:
            return len(lines)
        return 0