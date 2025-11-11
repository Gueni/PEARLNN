
## **Key Features of This Makefile**

### **Development Workflows**
```bash
make setup          # Complete development environment setup
make dev-start      # Start development session
make dev-test       # Run full test cycle
make pre-push       # Run all checks before pushing
```

### **Testing & Quality**
```bash
make test           # Run all tests
make test-cov       # Tests with coverage
make lint           # Run all linters
make format         # Auto-format code
```

### **Building & Distribution**
```bash
make build          # Build distribution packages
make release        # Create and upload release
make twine-check    # Validate package
```

### **Model Management**
```bash
make build-models   # Build initial models
make benchmark      # Run performance tests
make test-cli       # Test CLI functionality
```

### **Utility Commands**
```bash
make help           # Show all commands
make status         # Project status
make clean          # Clean all generated files
make data-clean     # Clean data directories
```

## 🚀 **Common Development Workflows**

### **1. New Contributor Setup**
```bash
git clone https://github.com/pearlnn/pearlnn
cd pearlnn
make setup
```

### **2. Daily Development**
```bash
make dev-start      # Start development
# ... make changes ...
make dev-test       # Test changes
make pre-push       # Final checks before push
```

### **3. Adding New Component**
```bash
make new-component NAME=capacitor
# Then implement the component in pearlnn/components/capacitor.py
```

### **4. Before Release**
```bash
make clean
make build
make twine-check
make test-cov
make release
```

This Makefile provides a comprehensive set of commands that make PEARLNN development efficient and consistent across different environments. It's especially useful for:

- **New contributors** getting started quickly
- **Maintainers** managing releases and quality
- **Developers** with consistent workflows
- **CI/CD pipelines** with standardized commands