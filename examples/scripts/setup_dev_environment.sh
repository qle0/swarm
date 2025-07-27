#!/bin/bash
# setup_dev_environment.sh - Настройка среды разработки для Swarm
set -e

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Функции для вывода
info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# Проверка операционной системы
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
        if command -v apt-get &> /dev/null; then
            DISTRO="debian"
        elif command -v yum &> /dev/null; then
            DISTRO="redhat"
        else
            DISTRO="unknown"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
        DISTRO="macos"
    else
        OS="unknown"
        DISTRO="unknown"
    fi
    
    info "Detected OS: $OS ($DISTRO)"
}

# Установка системных зависимостей
install_system_deps() {
    info "Installing system dependencies..."
    
    case $DISTRO in
        "debian")
            sudo apt update
            sudo apt install -y build-essential cmake git wget curl unzip
            sudo apt install -y python3.11 python3.11-dev python3.11-venv
            sudo apt install -y libgl1-mesa-glx mesa-utils xvfb
            ;;
        "macos")
            if ! command -v brew &> /dev/null; then
                info "Installing Homebrew..."
                /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            fi
            brew install python@3.11 cmake git wget
            ;;
        *)
            warning "Unknown distribution. Please install dependencies manually."
            ;;
    esac
    
    success "System dependencies installed"
}

# Создание Python окружения
setup_python_env() {
    info "Setting up Python environment..."
    
    # Создание виртуального окружения
    if [ ! -d "swarm_dev_env" ]; then
        python3.11 -m venv swarm_dev_env
        success "Virtual environment created"
    else
        info "Virtual environment already exists"
    fi
    
    # Активация окружения
    source swarm_dev_env/bin/activate
    
    # Обновление pip
    pip install --upgrade pip setuptools wheel
    
    # Установка основных зависимостей
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    else
        error "requirements.txt not found. Please run from Swarm repository root."
    fi
    
    # Установка дополнительных зависимостей для разработки
    pip install jupyter notebook tensorboard wandb
    pip install matplotlib seaborn plotly scikit-learn
    
    # Установка пакета в режиме разработки
    pip install -e .
    
    success "Python environment setup completed"
}

# Проверка установки
verify_installation() {
    info "Verifying installation..."
    
    # Активация окружения
    source swarm_dev_env/bin/activate
    
    # Проверка основных компонентов
    python -c "import pybullet; print('✓ PyBullet installed')" || error "PyBullet not installed"
    python -c "import stable_baselines3; print('✓ SB3 installed')" || error "Stable Baselines 3 not installed"
    python -c "import torch; print('✓ PyTorch installed')" || error "PyTorch not installed"
    python -c "import bittensor; print('✓ Bittensor installed')" || error "Bittensor not installed"
    
    success "Installation verification completed"
}

# Главная функция
main() {
    echo "🚀 Swarm Development Environment Setup"
    echo "======================================"
    
    # Проверка, что мы в корне репозитория
    if [ ! -f "setup.py" ] || [ ! -d "swarm" ]; then
        error "Please run this script from the Swarm repository root directory"
    fi
    
    detect_os
    install_system_deps
    setup_python_env
    verify_installation
    
    echo ""
    success "🎉 Development environment setup completed!"
    echo ""
    echo "To activate the environment:"
    echo "  source swarm_dev_env/bin/activate"
    echo ""
    echo "To start development:"
    echo "  cd examples/"
    echo "  python training/train_advanced.py --quick"
    echo ""
    echo "Happy coding! 🚁✨"
}

# Запуск главной функции
main "$@"