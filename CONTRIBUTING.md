
### `CONTRIBUTING.md`

```markdown
# Contributing to PEARLNN

🎉 **First off, thank you for considering contributing to PEARLNN!** 🎉

We're building a community-powered tool for electronics parameter fitting, and every contribution helps make it better for everyone. Whether you're an electronics engineer, data scientist, Python developer, or just passionate about open source, there are many ways to contribute.

## 🚀 Quick Start

### For New Contributors
1. **Fork** the repository on GitHub
2. **Clone** your fork locally
3. **Set up** the development environment (see below)
4. **Find an issue** labeled `good first issue` or `help wanted`
5. **Make your changes** and test them
6. **Submit a pull request**

### Development Environment Setup

```bash
# Clone your fork
git clone https://github.com/your-username/pearlnn
cd pearlnn

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .[dev]

# Set up pre-commit hooks
pre-commit install

# Run tests to verify setup
pytest
```

## 🎯 Ways to Contribute

### 1. 🐛 Report Bugs
If you find a bug, please create an issue with:

- **Description**: Clear description of the problem
- **Steps to Reproduce**: Step-by-step instructions
- **Expected vs Actual**: What you expected vs what happened
- **Environment**: OS, Python version, PEARLNN version
- **Screenshots/Logs**: If applicable

**Template**:
```markdown
## Bug Description

## Steps to Reproduce
1. 
2. 
3. 

## Expected Behavior

## Actual Behavior

## Environment
- OS: 
- Python Version:
- PEARLNN Version:
```

### 2. 💡 Suggest Enhancements
Have an idea for improvement? Create an issue with:

- **Clear description** of the feature
- **Use cases** and benefits
- **Any relevant examples** or mockups
- **Potential implementation** ideas (optional)

### 3. 🔧 Code Contributions
We welcome code contributions! Here's our process:

#### Finding Issues
- Look for issues labeled `good first issue` for beginner-friendly tasks
- `help wanted` issues need community assistance
- `bug` issues need fixing
- `enhancement` issues are feature requests

#### Making Changes
1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-description
   ```

2. **Make your changes** following our coding standards

3. **Write tests** for new functionality

4. **Update documentation** if needed

5. **Ensure all tests pass**:
   ```bash
   pytest
   ```

6. **Submit a pull request**

### 4. 📚 Documentation
Help us improve documentation:

- Fix typos and clarify explanations
- Add usage examples and tutorials
- Create component-specific guides
- Translate documentation to other languages
- Improve API documentation

### 5. 🔬 Add New Components
We're especially interested in contributions that add support for new electronic components!

#### Steps to Add a Component

1. **Create Component File** in `pearlnn/components/`:
   ```python
   from .base_component import BaseComponent
   
   class YourComponentAnalyzer(BaseComponent):
       def __init__(self):
           super().__init__("your_component")
           self.parameters = ["param1", "param2"]
           self.parameter_ranges = {
               "param1": (min_val, max_val),
           }
       
       def extract_parameters(self, features):
           # Implement parameter extraction
           pass
       
       def validate_parameters(self, parameters):
           # Implement validation
           pass
   ```

2. **Update Component Registry** in `pearlnn/components/__init__.py`

3. **Add Configuration** in `config/component_configs/`

4. **Write Tests** in `pearlnn/tests/test_components.py`

5. **Add Documentation** in `docs/component_guides/`

### 6. 🧪 Testing and Quality
- Write unit tests for new functionality
- Add integration tests
- Improve test coverage
- Report flaky tests
- Help with CI/CD improvements

### 7. 🌐 Community Support
- Help other users in discussions
- Answer questions on issues
- Share your success stories
- Spread the word about PEARLNN

## 🛠 Development Guidelines

### Code Style
We use several tools to maintain code quality:

```bash
# Format code
black pearlnn/ scripts/

# Check code style
flake8 pearlnn/ scripts/

# Type checking
mypy pearlnn/

# Sort imports
isort pearlnn/ scripts/
```

#### Python Style Guide
- Follow **PEP 8** guidelines
- Use **type hints** for all function signatures
- Write **docstrings** for all public functions and classes
- Use **descriptive variable names**
- Keep functions focused and small

#### Example Code Structure
```python
from typing import Dict, List, Optional
import numpy as np


class ExampleComponent:
    """Brief description of the component.
    
    Detailed explanation of what this component does and how it works.
    
    Args:
        param1: Description of parameter 1
        param2: Description of parameter 2
        
    Attributes:
        attribute1: Description of attribute 1
    """
    
    def __init__(self, param1: str, param2: Optional[int] = None):
        self.param1 = param1
        self.param2 = param2
        
    def process_data(self, data: np.ndarray) -> Dict[str, float]:
        """Process input data and return results.
        
        Args:
            data: Input data array with shape (n_samples, n_features)
            
        Returns:
            Dictionary containing processed results
            
        Raises:
            ValueError: If data format is invalid
        """
        if data.ndim != 2:
            raise ValueError("Data must be 2-dimensional")
        
        return {"result": 42.0}
```

### Testing
We use pytest for testing. All new code should include tests:

```python
import pytest
import numpy as np
from pearlnn.components.example_component import ExampleComponent


class TestExampleComponent:
    """Test suite for ExampleComponent."""
    
    def test_initialization(self):
        """Test component initialization."""
        component = ExampleComponent("test")
        assert component.param1 == "test"
        
    def test_process_data_valid(self):
        """Test data processing with valid input."""
        component = ExampleComponent("test")
        data = np.random.random((10, 5))
        result = component.process_data(data)
        assert "result" in result
        
    def test_process_data_invalid(self):
        """Test data processing with invalid input."""
        component = ExampleComponent("test")
        data = np.random.random(10)  # 1D array
        with pytest.raises(ValueError):
            component.process_data(data)
```

### Commit Messages
Use clear, descriptive commit messages following conventional commits:

- **feat**: New features
- **fix**: Bug fixes
- **docs**: Documentation changes
- **style**: Code style changes (formatting, etc.)
- **refactor**: Code refactoring
- **test**: Test-related changes
- **chore**: Maintenance tasks

**Examples**:
```
feat: add capacitor component analyzer
fix: resolve image processing memory leak
docs: update inductor analysis guide
test: add benchmark for feature extraction
```

### Pull Request Process

1. **Fork** the repository
2. **Create your feature branch** (`git checkout -b feature/amazing-feature`)
3. **Commit your changes** (`git commit -m 'feat: add amazing feature'`)
4. **Push to the branch** (`git push origin feature/amazing-feature`)
5. **Open a Pull Request**

#### PR Checklist
- [ ] Tests pass (`pytest`)
- [ ] Code follows style guidelines (`black`, `flake8`)
- [ ] Type checking passes (`mypy`)
- [ ] Documentation updated if needed
- [ ] Added tests for new functionality
- [ ] Commit messages follow conventions

#### PR Description Template
```markdown
## Description
Brief description of the changes

## Related Issues
Fixes #123

## Changes Made
- Change 1
- Change 2
- Change 3

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Manual testing performed

## Screenshots (if applicable)
```

## 🏗 Project Structure

```
pearlnn/
├── pearlnn/                 # Main package
│   ├── core/               # Core AI engine
│   ├── components/         # Electronic component analyzers
│   ├── data_processing/    # Input data handling
│   ├── community/          # Model sharing system
│   ├── training/           # Training utilities
│   ├── models/             # Neural network architectures
│   └── utils/              # Utility functions
├── config/                 # Configuration files
├── scripts/                # Development scripts
├── tests/                  # Test suite
└── docs/                   # Documentation
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=pearlnn

# Run specific test file
pytest pearlnn/tests/test_components.py

# Run tests with verbose output
pytest -v

# Run tests matching a pattern
pytest -k "test_inductor"
```

## 📖 Documentation

We use MkDocs for documentation. To build locally:

```bash
# Install documentation dependencies
pip install -e .[docs]

# Serve documentation locally
mkdocs serve

# Build documentation
mkdocs build
```

## 🐛 Debugging Tips

### Common Issues
1. **Import errors**: Make sure you installed in development mode (`pip install -e .`)
2. **Test failures**: Run `pytest -x` to stop on first failure
3. **Memory issues**: Use `PEARLNN_PERFORMANCE_MAX_MEMORY_USAGE=0.5` in `.env`

### Debug Mode
Enable debug mode in your `.env` file:
```env
PEARLNN_DEBUG_ENABLED=true
PEARLNN_DEBUG_VERBOSE_LOGGING=true
```

## 🌟 Recognition

All contributors will be recognized in:
- **Contributor list** in README
- **Release notes** for each version
- **Project documentation**
- **Community announcements**

## 📞 Getting Help

- **GitHub Discussions**: For questions and community support
- **GitHub Issues**: For bug reports and feature requests
- **Email**: community@pearlnn.dev

## 🏷 Code of Conduct

Please note that this project is released with a [Contributor Code of Conduct](CODE_OF_CONDUCT.md). By participating in this project you agree to abide by its terms.

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for helping build PEARLNN! Together, we're making electronics characterization accessible to everyone. 🚀

*"Alone we can do so little; together we can do so much."* - Helen Keller
```
