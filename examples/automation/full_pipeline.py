#!/usr/bin/env python3
"""
Полный автоматизированный пайплайн разработки и обучения для Swarm
"""
import os
import sys
import argparse
import time
import json
from pathlib import Path
from typing import Dict, Any

import yaml
import numpy as np
from stable_baselines3 import PPO


class SwarmDevelopmentPipeline:
    """Автоматизированный пайплайн разработки"""
    
    def __init__(self, config_path: str = None):
        self.config = self.load_config(config_path)
        self.results = {}
        self.start_time = time.time()
        
        # Создание директорий
        self.setup_directories()
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Загрузка конфигурации пайплайна"""
        
        default_config = {
            'pipeline': {
                'stages': ['train', 'test', 'optimize', 'validate'],
                'experiment_name': f'swarm_pipeline_{int(time.time())}',
                'base_dir': 'pipeline_results'
            },
            'training': {
                'phases': [
                    {
                        'name': 'quick_test',
                        'timesteps': 100000,
                        'n_envs': 2,
                        'learning_rate': 1e-3
                    },
                    {
                        'name': 'main_training',
                        'timesteps': 1000000,
                        'n_envs': 4,
                        'learning_rate': 3e-4
                    }
                ]
            },
            'testing': {
                'basic_episodes': 50,
                'robustness_episodes': 100
            },
            'optimization': {
                'max_size_mb': 10
            },
            'validation': {
                'success_threshold': 0.7,
                'robustness_threshold': 0.5
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                self._deep_update(default_config, user_config)
        
        return default_config
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """Глубокое обновление словаря"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def setup_directories(self):
        """Создание необходимых директорий"""
        
        base_dir = Path(self.config['pipeline']['base_dir'])
        experiment_name = self.config['pipeline']['experiment_name']
        
        self.experiment_dir = base_dir / experiment_name
        self.models_dir = self.experiment_dir / 'models'
        self.logs_dir = self.experiment_dir / 'logs'
        self.results_dir = self.experiment_dir / 'results'
        
        for directory in [self.experiment_dir, self.models_dir, self.logs_dir, self.results_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        print(f"Pipeline directory: {self.experiment_dir}")
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """Запуск полного пайплайна"""
        
        print(f"{'='*80}")
        print(f"SWARM DEVELOPMENT PIPELINE")
        print(f"Experiment: {self.config['pipeline']['experiment_name']}")
        print(f"{'='*80}")
        
        # Генерация отчета
        total_time = time.time() - self.start_time
        
        report = {
            'experiment_name': self.config['pipeline']['experiment_name'],
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_time_seconds': total_time,
            'config': self.config,
            'results': self.results
        }
        
        # Сохранение отчета
        report_path = self.experiment_dir / 'pipeline_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Pipeline setup completed!")
        print(f"Report saved to: {report_path}")
        
        return report


def main():
    parser = argparse.ArgumentParser(description='Swarm Development Pipeline')
    
    parser.add_argument('--config', type=str, 
                       help='Path to pipeline configuration file')
    parser.add_argument('--experiment-name', type=str,
                       help='Name of the experiment')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick test pipeline')
    
    args = parser.parse_args()
    
    # Создание пайплайна
    pipeline = SwarmDevelopmentPipeline(args.config)
    
    # Переопределение параметров
    if args.experiment_name:
        pipeline.config['pipeline']['experiment_name'] = args.experiment_name
    
    # Запуск пайплайна
    try:
        report = pipeline.run_full_pipeline()
        print("🎉 Pipeline completed successfully!")
        
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
    except Exception as e:
        print(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()