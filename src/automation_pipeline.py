"""
City Insights 360 - Automated Data Pipeline & Scheduling
======================================================

This module provides automated data refresh, validation, and scheduling
for the City Insights 360 urban analytics system.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime, timedelta
import schedule
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import warnings
import subprocess
import sys
from typing import Dict, List, Any
warnings.filterwarnings('ignore')

class AutomationPipeline:
    """Automated data pipeline with scheduling and monitoring"""
    
    def __init__(self, base_dir: str, config_path: str = None):
        self.base_dir = Path(base_dir)
        self.config_path = config_path or str(self.base_dir / "config" / "pipeline_config.json")
        self.log_dir = self.base_dir / "logs"
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Load configuration
        self.config = self._load_configuration()
        
        # Pipeline components
        self.src_dir = self.base_dir / "src"
        self.data_status = {}
        
    def _setup_logging(self):
        """Setup logging configuration"""
        log_file = self.log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(str(log_file)),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_configuration(self):
        """Load pipeline configuration"""
        default_config = {
            "schedule": {
                "data_refresh": "06:00",
                "analytics_update": "06:30", 
                "dashboard_refresh": "07:00",
                "health_check": "*/2",  # Every 2 hours
                "cleanup": "02:00"
            },
            "data_sources": {
                "air_quality": {"enabled": True, "priority": "high"},
                "traffic": {"enabled": True, "priority": "medium"},
                "demographics": {"enabled": True, "priority": "low"},
                "digital_infrastructure": {"enabled": True, "priority": "medium"}
            },
            "validation": {
                "min_records_threshold": 100,
                "max_null_percentage": 15.0,
                "max_processing_time": 3600  # 1 hour
            },
            "notifications": {
                "enabled": False,  # Set to True with email config
                "email": {
                    "smtp_server": "smtp.gmail.com",
                    "smtp_port": 587,
                    "sender_email": "",
                    "sender_password": "",
                    "recipients": []
                }
            },
            "retention": {
                "log_days": 30,
                "backup_days": 7,
                "temp_files_hours": 24
            }
        }
        
        try:
            config_file = Path(self.config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                    # Merge with defaults
                    default_config.update(user_config)
                    self.logger.info("Configuration loaded successfully")
            else:
                # Create default config file
                config_file.parent.mkdir(parents=True, exist_ok=True)
                with open(config_file, 'w') as f:
                    json.dump(default_config, f, indent=2)
                self.logger.info("Default configuration created")
        except Exception as e:
            self.logger.error(f"Error loading configuration: {str(e)}")
        
        return default_config
    
    def run_full_pipeline(self):
        """Execute complete data pipeline"""
        self.logger.info("🚀 Starting full pipeline execution")
        start_time = datetime.now()
        
        pipeline_status = {
            "start_time": start_time.isoformat(),
            "stages": {},
            "overall_status": "running"
        }
        
        try:
            # Stage 1: Data Catalog
            self.logger.info("📊 Stage 1: Data Cataloging")
            catalog_result = self._run_data_catalog()
            pipeline_status["stages"]["data_catalog"] = catalog_result
            
            # Stage 2: Data Integration
            self.logger.info("🔄 Stage 2: Data Integration")
            integration_result = self._run_data_integration()
            pipeline_status["stages"]["data_integration"] = integration_result
            
            # Stage 3: Analytics
            self.logger.info("🔍 Stage 3: Analytics Generation")
            analytics_result = self._run_analytics()
            pipeline_status["stages"]["analytics"] = analytics_result
            
            # Stage 4: Predictive Models
            self.logger.info("🤖 Stage 4: Predictive Modeling")
            modeling_result = self._run_predictive_models()
            pipeline_status["stages"]["predictive_modeling"] = modeling_result
            
            # Stage 5: Dashboard Preparation
            self.logger.info("📊 Stage 5: Dashboard Preparation")
            dashboard_result = self._run_dashboard_preparation()
            pipeline_status["stages"]["dashboard_preparation"] = dashboard_result
            
            # Final validation
            validation_result = self._validate_pipeline_output()
            pipeline_status["stages"]["validation"] = validation_result
            
            pipeline_status["overall_status"] = "success"
            self.logger.info("✅ Full pipeline completed successfully")
            
        except Exception as e:
            pipeline_status["overall_status"] = "failed"
            pipeline_status["error"] = str(e)
            self.logger.error(f"❌ Pipeline failed: {str(e)}")
        
        # Record completion
        end_time = datetime.now()
        pipeline_status["end_time"] = end_time.isoformat()
        pipeline_status["duration_minutes"] = (end_time - start_time).total_seconds() / 60
        
        # Save status
        self._save_pipeline_status(pipeline_status)
        
        # Send notifications if configured
        if self.config["notifications"]["enabled"]:
            self._send_notification(pipeline_status)
        
        return pipeline_status
    
    def _run_data_catalog(self):
        """Run data catalog module"""
        try:
            result = subprocess.run([
                sys.executable, 
                str(self.src_dir / "data_catalog.py")
            ], capture_output=True, text=True, cwd=str(self.base_dir))
            
            return {
                "status": "success" if result.returncode == 0 else "failed",
                "return_code": result.returncode,
                "output": result.stdout[-500:] if result.stdout else "",
                "error": result.stderr[-500:] if result.stderr else ""
            }
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    
    def _run_data_integration(self):
        """Run data integration module"""
        try:
            result = subprocess.run([
                sys.executable,
                str(self.src_dir / "data_integration.py")
            ], capture_output=True, text=True, cwd=str(self.base_dir))
            
            return {
                "status": "success" if result.returncode == 0 else "failed",
                "return_code": result.returncode,
                "output": result.stdout[-500:] if result.stdout else "",
                "error": result.stderr[-500:] if result.stderr else ""
            }
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    
    def _run_analytics(self):
        """Run analytics framework"""
        try:
            result = subprocess.run([
                sys.executable,
                str(self.src_dir / "analytics_framework.py")
            ], capture_output=True, text=True, cwd=str(self.base_dir))
            
            return {
                "status": "success" if result.returncode == 0 else "failed",
                "return_code": result.returncode,
                "output": result.stdout[-500:] if result.stdout else "",
                "error": result.stderr[-500:] if result.stderr else ""
            }
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    
    def _run_predictive_models(self):
        """Run predictive modeling"""
        try:
            result = subprocess.run([
                sys.executable,
                str(self.src_dir / "predictive_models.py")
            ], capture_output=True, text=True, cwd=str(self.base_dir))
            
            return {
                "status": "success" if result.returncode == 0 else "failed",
                "return_code": result.returncode,
                "output": result.stdout[-500:] if result.stdout else "",
                "error": result.stderr[-500:] if result.stderr else ""
            }
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    
    def _run_dashboard_preparation(self):
        """Run Power BI dashboard preparation"""
        try:
            result = subprocess.run([
                sys.executable,
                str(self.src_dir / "powerbi_dashboard.py")
            ], capture_output=True, text=True, cwd=str(self.base_dir))
            
            return {
                "status": "success" if result.returncode == 0 else "failed",
                "return_code": result.returncode,
                "output": result.stdout[-500:] if result.stdout else "",
                "error": result.stderr[-500:] if result.stderr else ""
            }
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    
    def _validate_pipeline_output(self):
        """Validate pipeline output quality"""
        validation_results = {}
        
        # Check integrated data
        integrated_data_dir = self.base_dir / "integrated_data"
        if integrated_data_dir.exists():
            validation_results["integrated_data"] = self._validate_data_quality(integrated_data_dir)
        
        # Check analytics output
        analytics_dir = self.base_dir / "analytics_output"
        if analytics_dir.exists():
            validation_results["analytics"] = self._validate_analytics_output(analytics_dir)
        
        # Check dashboard data
        dashboard_dir = self.base_dir / "powerbi_dashboard"
        if dashboard_dir.exists():
            validation_results["dashboard_data"] = self._validate_dashboard_data(dashboard_dir)
        
        return validation_results
    
    def _validate_data_quality(self, data_dir: Path):
        """Validate integrated data quality"""
        issues = []
        metrics = {"total_files": 0, "total_records": 0, "empty_files": 0}
        
        for csv_file in data_dir.glob("*.csv"):
            try:
                df = pd.read_csv(csv_file)
                metrics["total_files"] += 1
                metrics["total_records"] += len(df)
                
                if len(df) == 0:
                    metrics["empty_files"] += 1
                    issues.append(f"Empty file: {csv_file.name}")
                
                # Check null percentages
                null_pct = (df.isnull().sum().sum() / df.size) * 100
                if null_pct > self.config["validation"]["max_null_percentage"]:
                    issues.append(f"High null percentage in {csv_file.name}: {null_pct:.1f}%")
                
                # Check minimum records
                if len(df) < self.config["validation"]["min_records_threshold"]:
                    issues.append(f"Low record count in {csv_file.name}: {len(df)} records")
                    
            except Exception as e:
                issues.append(f"Error reading {csv_file.name}: {str(e)}")
        
        return {
            "status": "passed" if len(issues) == 0 else "warning",
            "metrics": metrics,
            "issues": issues
        }
    
    def _validate_analytics_output(self, analytics_dir: Path):
        """Validate analytics output"""
        insights_file = analytics_dir / "analytics_insights.json"
        
        if not insights_file.exists():
            return {"status": "failed", "error": "Analytics insights file missing"}
        
        try:
            with open(insights_file, 'r') as f:
                insights = json.load(f)
            
            required_sections = [
                "air_quality_analysis",
                "demographic_trends", 
                "city_rankings",
                "executive_summary"
            ]
            
            missing_sections = [section for section in required_sections 
                              if section not in insights]
            
            return {
                "status": "passed" if len(missing_sections) == 0 else "warning",
                "sections_found": len(insights),
                "missing_sections": missing_sections
            }
            
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    
    def _validate_dashboard_data(self, dashboard_dir: Path):
        """Validate Power BI dashboard data"""
        required_files = [
            "executive_summary.csv",
            "air_quality_kpis.csv", 
            "city_rankings.csv",
            "dashboard_specification.json"
        ]
        
        missing_files = []
        file_metrics = {}
        
        for filename in required_files:
            file_path = dashboard_dir / filename
            if not file_path.exists():
                missing_files.append(filename)
            elif filename.endswith('.csv'):
                try:
                    df = pd.read_csv(file_path)
                    file_metrics[filename] = len(df)
                except:
                    file_metrics[filename] = "error"
        
        return {
            "status": "passed" if len(missing_files) == 0 else "failed",
            "missing_files": missing_files,
            "file_metrics": file_metrics
        }
    
    def _save_pipeline_status(self, status: Dict):
        """Save pipeline execution status"""
        status_file = self.log_dir / f"pipeline_status_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(status_file, 'w') as f:
            json.dump(status, f, indent=2, default=str)
        
        # Also save as latest status
        latest_status_file = self.log_dir / "latest_pipeline_status.json"
        with open(latest_status_file, 'w') as f:
            json.dump(status, f, indent=2, default=str)
    
    def _send_notification(self, status: Dict):
        """Send email notification about pipeline status"""
        if not self.config["notifications"]["email"]["recipients"]:
            return
        
        try:
            subject = f"City Insights 360 - Pipeline {status['overall_status'].title()}"
            
            body = f"""
City Insights 360 Pipeline Report
================================

Status: {status['overall_status'].upper()}
Start Time: {status['start_time']}
Duration: {status.get('duration_minutes', 0):.1f} minutes

Stage Results:
"""
            
            for stage, result in status.get('stages', {}).items():
                body += f"- {stage}: {result.get('status', 'unknown')}\n"
            
            if status['overall_status'] == 'failed':
                body += f"\nError: {status.get('error', 'Unknown error')}\n"
            
            # Send email
            self._send_email(subject, body)
            
        except Exception as e:
            self.logger.error(f"Failed to send notification: {str(e)}")
    
    def _send_email(self, subject: str, body: str):
        """Send email notification"""
        email_config = self.config["notifications"]["email"]
        
        msg = MIMEMultipart()
        msg['From'] = email_config["sender_email"]
        msg['To'] = ", ".join(email_config["recipients"])
        msg['Subject'] = subject
        
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP(email_config["smtp_server"], email_config["smtp_port"])
        server.starttls()
        server.login(email_config["sender_email"], email_config["sender_password"])
        
        text = msg.as_string()
        server.sendmail(email_config["sender_email"], email_config["recipients"], text)
        server.quit()
    
    def run_health_check(self):
        """Run system health check"""
        self.logger.info("🏥 Running health check")
        
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "checks": {}
        }
        
        # Check disk space
        health_status["checks"]["disk_space"] = self._check_disk_space()
        
        # Check data freshness
        health_status["checks"]["data_freshness"] = self._check_data_freshness()
        
        # Check log file size
        health_status["checks"]["log_files"] = self._check_log_files()
        
        # Save health status
        health_file = self.log_dir / "health_check.json"
        with open(health_file, 'w') as f:
            json.dump(health_status, f, indent=2, default=str)
        
        return health_status
    
    def _check_disk_space(self):
        """Check available disk space"""
        try:
            import shutil
            total, used, free = shutil.disk_usage(str(self.base_dir))
            free_gb = free // (1024**3)
            
            return {
                "status": "good" if free_gb > 5 else "warning",
                "free_space_gb": free_gb,
                "message": f"{free_gb} GB available"
            }
        except:
            return {"status": "unknown", "message": "Could not check disk space"}
    
    def _check_data_freshness(self):
        """Check data freshness"""
        integrated_data_dir = self.base_dir / "integrated_data"
        
        if not integrated_data_dir.exists():
            return {"status": "warning", "message": "No integrated data found"}
        
        try:
            # Check modification time of integrated data
            csv_files = list(integrated_data_dir.glob("*.csv"))
            if not csv_files:
                return {"status": "warning", "message": "No data files found"}
            
            latest_mod_time = max(file.stat().st_mtime for file in csv_files)
            hours_since_update = (time.time() - latest_mod_time) / 3600
            
            if hours_since_update < 24:
                status = "good"
            elif hours_since_update < 48:
                status = "warning"
            else:
                status = "stale"
            
            return {
                "status": status,
                "hours_since_update": round(hours_since_update, 1),
                "message": f"Data last updated {hours_since_update:.1f} hours ago"
            }
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _check_log_files(self):
        """Check log file sizes"""
        try:
            total_log_size = sum(
                f.stat().st_size for f in self.log_dir.glob("*.log")
            ) / (1024**2)  # MB
            
            return {
                "status": "good" if total_log_size < 100 else "warning",
                "total_size_mb": round(total_log_size, 1),
                "message": f"{total_log_size:.1f} MB of logs"
            }
        except:
            return {"status": "unknown", "message": "Could not check log files"}
    
    def cleanup_old_files(self):
        """Clean up old log files and temporary data"""
        self.logger.info("🧹 Running cleanup")
        
        cleanup_stats = {"deleted_files": 0, "freed_space_mb": 0}
        
        try:
            # Clean old log files
            cutoff_date = datetime.now() - timedelta(days=self.config["retention"]["log_days"])
            
            for log_file in self.log_dir.glob("*.log"):
                if datetime.fromtimestamp(log_file.stat().st_mtime) < cutoff_date:
                    size_mb = log_file.stat().st_size / (1024**2)
                    log_file.unlink()
                    cleanup_stats["deleted_files"] += 1
                    cleanup_stats["freed_space_mb"] += size_mb
            
            self.logger.info(f"Cleanup completed: {cleanup_stats}")
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {str(e)}")
        
        return cleanup_stats
    
    def setup_scheduler(self):
        """Setup scheduled tasks"""
        self.logger.info("⏰ Setting up scheduler")
        
        # Daily full pipeline
        schedule.every().day.at(self.config["schedule"]["data_refresh"]).do(self.run_full_pipeline)
        
        # Health checks every 2 hours
        schedule.every(2).hours.do(self.run_health_check)
        
        # Daily cleanup
        schedule.every().day.at(self.config["schedule"]["cleanup"]).do(self.cleanup_old_files)
        
        self.logger.info("Scheduler configured")
    
    def start_scheduler(self):
        """Start the scheduler daemon"""
        self.logger.info("🚀 Starting scheduler daemon")
        
        self.setup_scheduler()
        
        # Run initial health check
        self.run_health_check()
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            self.logger.info("Scheduler stopped by user")
        except Exception as e:
            self.logger.error(f"Scheduler error: {str(e)}")

def create_scheduler_service():
    """Create a Windows service script for the scheduler"""
    service_script = '''
import sys
import os
sys.path.append(os.path.dirname(__file__))

from automation_pipeline import AutomationPipeline

if __name__ == "__main__":
    pipeline = AutomationPipeline(r"C:\\Users\\91892\\OneDrive\\Desktop\\City Insights 360")
    pipeline.start_scheduler()
'''
    
    base_dir = Path(r"C:\Users\91892\OneDrive\Desktop\City Insights 360")
    service_file = base_dir / "src" / "scheduler_service.py"
    
    with open(service_file, 'w') as f:
        f.write(service_script)
    
    return service_file

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="City Insights 360 Automation Pipeline")
    parser.add_argument("--mode", choices=["run", "schedule", "health", "cleanup"], 
                       default="run", help="Operation mode")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    base_dir = r"C:\Users\91892\OneDrive\Desktop\City Insights 360"
    pipeline = AutomationPipeline(base_dir)
    
    if args.mode == "run":
        # Run full pipeline once
        result = pipeline.run_full_pipeline()
        print(f"Pipeline completed with status: {result['overall_status']}")
        
    elif args.mode == "schedule":
        # Start scheduler daemon
        pipeline.start_scheduler()
        
    elif args.mode == "health":
        # Run health check
        health = pipeline.run_health_check()
        print("Health check completed")
        
    elif args.mode == "cleanup":
        # Run cleanup
        cleanup = pipeline.cleanup_old_files()
        print(f"Cleanup completed: {cleanup}")
    
    # Create scheduler service file
    service_file = create_scheduler_service()
    print(f"📁 Scheduler service created: {service_file}")