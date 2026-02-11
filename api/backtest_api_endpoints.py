# Add these imports at the top of main.py
from api.backtest_optimizer import BacktestOptimizer, run_optimization_job
import threading
import uuid

# Global job storage (in production, use Redis or database)
_optimization_jobs = {}

class OptimizationRequest(BaseModel):
    wave_values: List[float] = [0.7, 0.8, 0.9]
    target_values: List[int] = [5, 10, 15]
    stoploss_values: List[int] = [3, 5, 7]
    model: str = "KING 👑.pkl"
    council_filter: Optional[str] = None
    exchange: str = "CRYPTO"
    timeframe: str = "1H"
    start_date: str
    end_date: str
    capital: int = 100000

@app.post("/api/backtest/optimize")
async def start_optimization(req: OptimizationRequest):
    """
    Start a batch optimization job.
    Returns job_id for tracking progress.
    """
    job_id = str(uuid.uuid4())
    
    # Prepare job parameters
    job_params = {
        "wave_values": req.wave_values,
        "target_values": req.target_values,
        "stoploss_values": req.stoploss_values,
        "base_params": {
            "model": req.model,
            "council_filter": req.council_filter,
            "exchange": req.exchange,
            "timeframe": req.timeframe,
            "start_date": req.start_date,
            "end_date": req.end_date,
            "capital": req.capital
        }
    }
    
    # Initialize job status
    total_combinations = len(req.wave_values) * len(req.target_values) * len(req.stoploss_values)
    _optimization_jobs[job_id] = {
        "status": "running",
        "progress": 0,
        "total": total_combinations,
        "results": [],
        "started_at": datetime.now().isoformat(),
        "completed_at": None,
        "error": None
    }
    
    # Run optimization in background thread
    def run_job():
        try:
            optimizer = BacktestOptimizer()
            
            def progress_callback(current, total, result):
                _optimization_jobs[job_id]["progress"] = current
                if result:
                    _optimization_jobs[job_id]["results"].append(result)
            
            results_df = optimizer.optimize_parameters(
                wave_values=req.wave_values,
                target_values=req.target_values,
                stoploss_values=req.stoploss_values,
                base_params=job_params["base_params"],
                progress_callback=progress_callback
            )
            
            # Save results
            csv_path = optimizer.save_results(results_df, f"optimization_{job_id}.csv")
            report = optimizer.generate_report(results_df)
            report_path = csv_path.replace('.csv', '_report.txt')
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            
            # Update job status
            _optimization_jobs[job_id].update({
                "status": "completed",
                "completed_at": datetime.now().isoformat(),
                "csv_path": csv_path,
                "report_path": report_path,
                "results_df": results_df.to_dict('records')
            })
            
        except Exception as e:
            _optimization_jobs[job_id].update({
                "status": "failed",
                "error": str(e),
                "completed_at": datetime.now().isoformat()
            })
    
    thread = threading.Thread(target=run_job, daemon=True)
    thread.start()
    
    return {
        "job_id": job_id,
        "status": "started",
        "total_tests": total_combinations
    }

@app.get("/api/backtest/results/{job_id}")
async def get_optimization_results(job_id: str):
    """
    Get optimization job status and results.
    """
    if job_id not in _optimization_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = _optimization_jobs[job_id]
    
    return {
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "total": job["total"],
        "started_at": job["started_at"],
        "completed_at": job["completed_at"],
        "results": job.get("results_df", []),
        "error": job.get("error")
    }

@app.get("/api/backtest/export/{job_id}")
async def export_optimization_results(job_id: str, format: str = "csv"):
    """
    Export optimization results as CSV or report.
    """
    if job_id not in _optimization_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = _optimization_jobs[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    if format == "csv":
        file_path = job.get("csv_path")
    elif format == "report":
        file_path = job.get("report_path")
    else:
        raise HTTPException(status_code=400, detail="Invalid format. Use 'csv' or 'report'")
    
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    from fastapi.responses import FileResponse
    return FileResponse(
        path=file_path,
        filename=os.path.basename(file_path),
        media_type='text/csv' if format == "csv" else 'text/plain'
    )
