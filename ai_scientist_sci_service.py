from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from enum import Enum
import uuid
import asyncio
from datetime import datetime

app = FastAPI(title="AI Scientist SCI Reconstruction Service", version="1.0.0")

active_token = "nWpNRwtHETYvUAbdwbIlsTjGzdjAQcvqfaQObcGeKLp"

class ForwardConfig(BaseModel):
    compression_ratio: int = Field(default=8, description="Compression ratio")
    mask_type: str = Field(default="random", description="Mask type")
    noise_level: float = Field(default=0.01, description="Noise level")


class TrainRequest(BaseModel):
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    forward_config: ForwardConfig
    recon_family: str
    uq_scheme: str
    recon_params: Dict[str, Any]
    uq_params: Dict[str, Any]
    train_config: Dict[str, Any]


class TrainResponse(BaseModel):
    job_id: str
    status: str
    message: str
    checkpoint_id: Optional[str] = None
    training_time: Optional[float] = None

class MetricsResult(BaseModel):
    psnr: float
    ssim: float
    rmse: float
    coverage: float
    latency: float
    calibration_error: Optional[float] = None

class EvaluateResponse(BaseModel):
    job_id: str
    status: str
    train_log: str
    metrics: Optional[MetricsResult] = None
    detailed_results: Optional[Dict[str, Any]] = None
    evaluation_time: Optional[float] = None

# ============ 任务存储 ============
jobs_store = {}

# ============ API端点 ============

@app.post("/api/v1/train", response_model=TrainResponse)
async def train_reconstructor(
    request: TrainRequest,
    background_tasks: BackgroundTasks
):
    job_id = request.job_id

    # simply validate token
    auth_header = request.headers.get("Authorization")
    if auth_header != f"Bearer {active_token}":
        raise HTTPException(status_code=401, detail="Invalid token")

    # 初始化任务状态
    jobs_store[job_id] = {
        "status": "pending",
        "type": "training",
        "created_at": datetime.now().isoformat(),
        "request": request.dict()
    }

    # 异步执行训练任务
    background_tasks.add_task(
        _execute_training,
        job_id,
        request
    )

    return TrainResponse(
        job_id=job_id,
        status="accepted",
        message=f"Training job submitted, reconstruction family: {request.recon_family}"
    )

@app.get("/api/v1/jobs/{job_id}", response_model=EvaluateResponse)
async def get_by_job(job_id: str):
    """Get job results or status"""
    if job_id not in jobs_store:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs_store[job_id]

    # If job is completed, populate metrics
    if job["status"] == "completed":
        # Extract metrics if available, otherwise use mock data
        metrics = MetricsResult(
            psnr=job.get("metrics", {}).get("psnr", 35.5),
            ssim=job.get("metrics", {}).get("ssim", 0.95),
            rmse=job.get("metrics", {}).get("rmse", 0.012),
            coverage=job.get("metrics", {}).get("coverage", 0.98),
            latency=job.get("metrics", {}).get("latency", 0.15),
            calibration_error=job.get("metrics", {}).get("calibration_error")
        )

        return EvaluateResponse(
            job_id=job_id,
            status="completed",
            metrics=metrics,
            detailed_results=job.get("detailed_results"),
            evaluation_time=job.get("training_time")
        )

    # For non-completed jobs, return EvaluateResponse with metrics=None
    return EvaluateResponse(
        job_id=job_id,
        status=job["status"],
        metrics=None,
        detailed_results={"error": job["error"]} if job.get("error") else None,
        evaluation_time=None
    )

@app.get("/api/v1/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "active_jobs": len([j for j in jobs_store.values() if j["status"] == "running"]),
        "timestamp": datetime.now().isoformat()
    }

# ============ 后台任务执行函数 ============

async def _execute_training(job_id: str, request: TrainRequest):
    """执行训练任务的实际逻辑"""
    try:
        jobs_store[job_id]["status"] = "running"
        jobs_store[job_id]["started_at"] = datetime.now().isoformat()

        await asyncio.sleep(2)  # The actual training will take longer!

        checkpoint_id = f"checkpoint_{job_id[:8]}"

        # 更新任务状态
        jobs_store[job_id].update({
            "status": "completed",
            "checkpoint_id": checkpoint_id,
            "training_time": 120.5,  # 模拟训练时间
            "completed_at": datetime.now().isoformat(),
            "final_loss": 0.0023
        })

    except Exception as e:
        jobs_store[job_id].update({
            "status": "failed",
            "error": str(e),
            "failed_at": datetime.now().isoformat()
        })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
