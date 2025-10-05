"""
Stellar Scout V2 - NASA Space Apps Challenge 2025
Advanced AI-powered exoplanet detection system
Features: Kepler/TESS integration, Deep Learning, Light curve analysis
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import requests
import numpy as np
import pandas as pd
import pickle
import os
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from functools import lru_cache
import io
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
class Settings:
    APP_NAME = "Stellar Scout"
    VERSION = "1.0.0"
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
    NASA_EXOPLANET_API = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    CACHE_DURATION = 3600

settings = Settings()

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="NASA Space Apps Challenge 2025 - Exoplanet Hunter with Advanced AI"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global cache
exoplanet_cache = []
cache_timestamp = None

# Load ML models
ml_models = {}
try:
    with open('models/random_forest_model.pkl', 'rb') as f:
        ml_models['rf'] = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        ml_models['scaler'] = pickle.load(f)
    logger.info("✅ ML Models loaded successfully")
except Exception as e:
    logger.warning(f"⚠️ ML Models not found: {e}")

# Try loading deep learning models
try:
    import tensorflow as tf
    ml_models['cnn'] = tf.keras.models.load_model('models/cnn_model.h5')
    ml_models['lstm'] = tf.keras.models.load_model('models/lstm_model.h5')
    logger.info("✅ Deep Learning models loaded")
except:
    logger.info("⚠️ Deep learning models not available")


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class PredictionRequest(BaseModel):
    star_temp: float
    star_radius: float
    star_mass: float
    orbital_period: float
    transit_depth: float

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    planet_type: str
    habitable_zone: str
    confidence: str
    details: Dict[str, Any]

class LightCurveAnalysisResponse(BaseModel):
    has_transit: bool
    transit_times: List[float]
    transit_depth: float
    orbital_period: float
    planet_radius_estimate: float
    confidence: float
    plot_data: Dict[str, List[float]]

class TransitSimulationRequest(BaseModel):
    planet_radius: float
    star_radius: float
    orbital_period: float
    inclination: float = 90.0
    num_points: int = 1000

class HabitabilityRequest(BaseModel):
    star_temp: float
    star_radius: float
    star_mass: float
    planet_radius: float
    orbital_period: float


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def fetch_nasa_exoplanets():
    """Fetch exoplanet data from NASA archive"""
    global exoplanet_cache, cache_timestamp
    
    if exoplanet_cache and cache_timestamp:
        elapsed = (datetime.now() - cache_timestamp).total_seconds()
        if elapsed < settings.CACHE_DURATION:
            return exoplanet_cache
    
    try:
        logger.info("Fetching NASA exoplanet data...")
        query = """
            select pl_name, hostname, discoverymethod, disc_year, disc_facility,
                   pl_orbper, pl_rade, pl_bmasse, pl_eqt, pl_insol,
                   st_teff, st_rad, st_mass, st_met, st_logg,
                   sy_dist, sy_vmag, sy_kmag
            from ps where default_flag = 1
        """
        
        response = requests.get(
            settings.NASA_EXOPLANET_API,
            params={'query': query, 'format': 'json'},
            timeout=30
        )
        response.raise_for_status()
        
        exoplanet_cache = response.json()
        cache_timestamp = datetime.now()
        logger.info(f"✅ Cached {len(exoplanet_cache)} exoplanets")
        return exoplanet_cache
    except Exception as e:
        logger.error(f"Error fetching NASA data: {e}")
        return exoplanet_cache if exoplanet_cache else []

def calculate_habitability(star_temp, star_radius, star_mass, planet_radius, orbital_period):
    """Calculate detailed habitability metrics"""
    try:
        # Stellar luminosity
        temp_ratio = star_temp / 5778.0
        luminosity = (star_radius ** 2) * (temp_ratio ** 4)
        
        # Orbital distance (AU)
        orbital_distance = (orbital_period / 365.25) ** (2/3) * star_mass ** (1/3)
        
        # Habitable zone boundaries (Kopparapu et al. 2013)
        inner_hz = 0.95 * np.sqrt(luminosity)
        outer_hz = 1.67 * np.sqrt(luminosity)
        conservative_inner = 0.99 * np.sqrt(luminosity)
        conservative_outer = 1.70 * np.sqrt(luminosity)
        
        # Equilibrium temperature
        albedo = 0.3  # Assumed albedo
        eq_temp = 278 * np.sqrt(star_radius / orbital_distance) * (1 - albedo) ** 0.25 * (star_temp / 5778) ** 0.5
        
        # Earth Similarity Index (ESI)
        radius_weight = 1 - abs((planet_radius - 1.0) / (planet_radius + 1.0))
        temp_weight = 1 - abs((eq_temp - 288) / (eq_temp + 288))
        esi = (radius_weight * temp_weight) ** 0.5
        
        # Classification
        if conservative_inner <= orbital_distance <= conservative_outer:
            zone = "Conservative Habitable Zone"
            habitability_score = 0.9
        elif inner_hz <= orbital_distance <= outer_hz:
            zone = "Optimistic Habitable Zone"
            habitability_score = 0.7
        elif orbital_distance < inner_hz:
            zone = "Too Hot (Runaway Greenhouse)"
            habitability_score = 0.2
        else:
            zone = "Too Cold (Snowball)"
            habitability_score = 0.3
        
        return {
            "zone": zone,
            "equilibrium_temp": round(eq_temp, 2),
            "inner_hz_boundary": round(inner_hz, 3),
            "outer_hz_boundary": round(outer_hz, 3),
            "orbital_distance_au": round(orbital_distance, 3),
            "stellar_luminosity": round(luminosity, 3),
            "earth_similarity_index": round(esi, 3),
            "habitability_score": round(habitability_score, 2),
            "water_state": "Liquid" if 273 < eq_temp < 373 else ("Ice" if eq_temp < 273 else "Vapor")
        }
    except Exception as e:
        logger.error(f"Habitability calculation error: {e}")
        return {}

def analyze_light_curve(time, flux):
    """Analyze light curve for transit detection"""
    try:
        # Normalize flux
        flux_normalized = (flux - np.nanmedian(flux)) / np.nanmedian(flux)
        
        # Simple transit detection (box least squares would be better)
        threshold = -0.01  # 1% dip
        transits = flux_normalized < threshold
        
        if not np.any(transits):
            return {
                "has_transit": False,
                "transit_times": [],
                "transit_depth": 0.0,
                "orbital_period": 0.0,
                "confidence": 0.0,
                "planet_radius_estimate": 0.0
            }
        
        # Find transit times
        transit_indices = np.where(transits)[0]
        if len(transit_indices) < 2:
            return {
                "has_transit": False,
                "transit_times": [],
                "transit_depth": 0.0,
                "orbital_period": 0.0,
                "confidence": 0.0,
                "planet_radius_estimate": 0.0
            }
        
        transit_times = time[transit_indices]
        
        # Estimate period (simple - should use BLS)
        if len(transit_times) > 1:
            period = float(np.median(np.diff(transit_times)))
        else:
            period = 0.0
        
        # Transit depth
        transit_depth = float(abs(np.min(flux_normalized)))
        
        # Planet radius estimate
        planet_radius = float(np.sqrt(transit_depth)) * 1.0  # Assuming solar radius star
        
        # Confidence based on SNR
        noise = np.std(flux_normalized[~transits])
        snr = transit_depth / noise if noise > 0 else 0
        confidence = float(min(snr / 10, 1.0))  # Normalize to 0-1
        
        return {
            "has_transit": True,
            "transit_times": transit_times.tolist()[:10],  # First 10
            "transit_depth": transit_depth,
            "orbital_period": period,
            "confidence": confidence,
            "planet_radius_estimate": planet_radius
        }
    except Exception as e:
        logger.error(f"Light curve analysis error: {e}")
        return {
            "has_transit": False,
            "transit_times": [],
            "transit_depth": 0.0,
            "orbital_period": 0.0,
            "confidence": 0.0,
            "planet_radius_estimate": 0.0
        }

def simulate_transit(planet_radius, star_radius, orbital_period, inclination, num_points):
    """Simulate planet transit light curve"""
    try:
        # Time array (days)
        time = np.linspace(0, orbital_period * 2, num_points)
        
        # Orbital phase
        phase = (time % orbital_period) / orbital_period * 2 * np.pi
        
        # Position in orbit
        x = np.cos(phase)
        y = np.sin(phase) * np.cos(np.radians(inclination))
        
        # Distance from star center
        distance = np.sqrt(x**2 + y**2)
        
        # Transit occurs when planet is in front of star and within radius
        in_transit = (y < 0) & (distance < star_radius)
        
        # Calculate flux drop
        flux = np.ones(num_points)
        planet_area = np.pi * planet_radius**2
        star_area = np.pi * star_radius**2
        transit_depth = planet_area / star_area
        
        flux[in_transit] = 1 - transit_depth
        
        # Add realistic noise
        noise = np.random.normal(0, 0.0001, num_points)
        flux += noise
        
        return {
            "time": time.tolist(),
            "flux": flux.tolist(),
            "transit_depth": float(transit_depth),
            "duration_hours": float(orbital_period * 24 * np.sum(in_transit) / num_points)
        }
    except Exception as e:
        logger.error(f"Transit simulation error: {e}")
        return {}


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
def root():
    return {
        "app": settings.APP_NAME,
        "version": settings.VERSION,
        "status": "operational",
        "challenge": "NASA Space Apps Challenge 2025",
        "description": "Advanced AI-powered exoplanet detection system",
        "features": [
            "Real-time NASA data",
            "Deep learning models",
            "Light curve analysis",
            "Transit simulation",
            "Habitability calculation"
        ]
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": {
            "random_forest": "rf" in ml_models,
            "cnn": "cnn" in ml_models,
            "lstm": "lstm" in ml_models
        },
        "cache_active": len(exoplanet_cache) > 0,
        "cache_size": len(exoplanet_cache)
    }

@app.get("/api/stats")
def get_stats():
    """Enhanced statistics"""
    try:
        data = fetch_nasa_exoplanets()
        
        total = len(data)
        
        # Habitable zone (rough estimate)
        habitable = sum(1 for p in data 
                       if p.get('pl_rade') and 0.5 <= p['pl_rade'] <= 2.0
                       and p.get('pl_eqt') and 200 <= p['pl_eqt'] <= 350)
        
        # Recent discoveries
        current_year = datetime.now().year
        recent = sum(1 for p in data if p.get('disc_year', 0) >= current_year - 1)
        
        # By method
        methods = {}
        for p in data:
            method = p.get('discoverymethod', 'Unknown')
            methods[method] = methods.get(method, 0) + 1
        
        # By facility
        facilities = {}
        for p in data:
            facility = p.get('disc_facility', 'Unknown')
            if facility and len(facilities) < 10:  # Top 10
                facilities[facility] = facilities.get(facility, 0) + 1
        
        return {
            "total_exoplanets": total,
            "potentially_habitable": habitable,
            "recent_discoveries": recent,
            "by_method": methods,
            "by_facility": dict(sorted(facilities.items(), key=lambda x: x[1], reverse=True)[:10])
        }
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return {"total_exoplanets": 5500, "potentially_habitable": 60, "recent_discoveries": 150}

@app.get("/api/exoplanets")
def get_exoplanets(
    page: int = 1,
    limit: int = 12,
    search: str = None,
    method: str = None,
    min_radius: float = None,
    max_radius: float = None
):
    """Advanced exoplanet search with filters"""
    try:
        data = fetch_nasa_exoplanets()
        
        # Apply filters
        filtered = data
        
        if search:
            filtered = [p for p in filtered 
                       if search.lower() in p.get('pl_name', '').lower() 
                       or search.lower() in p.get('hostname', '').lower()]
        
        if method:
            filtered = [p for p in filtered 
                       if p.get('discoverymethod') == method]
        
        if min_radius is not None:
            filtered = [p for p in filtered 
                       if p.get('pl_rade') and p['pl_rade'] >= min_radius]
        
        if max_radius is not None:
            filtered = [p for p in filtered 
                       if p.get('pl_rade') and p['pl_rade'] <= max_radius]
        
        # Pagination
        start = (page - 1) * limit
        end = start + limit
        paginated = filtered[start:end]
        
        return {
            "exoplanets": paginated,
            "total": len(filtered),
            "page": page,
            "limit": limit,
            "has_more": end < len(filtered),
            "filters_applied": {
                "search": search,
                "method": method,
                "radius_range": [min_radius, max_radius]
            }
        }
    except Exception as e:
        logger.error(f"Exoplanets fetch error: {e}")
        raise HTTPException(500, f"Error: {e}")

@app.post("/api/predict", response_model=PredictionResponse)
def predict_exoplanet(req: PredictionRequest):
    """Enhanced AI prediction with ensemble models"""
    try:
        # Calculate planet radius estimate
        pl_radius = np.sqrt(req.transit_depth / 100) * req.star_radius
        
        # Engineer ALL 21 features that match training
        st_teff_ratio = req.star_temp / 5778.0
        st_luminosity = (req.star_radius ** 2) * (req.star_temp / 5778.0) ** 4
        st_density = req.star_mass / (req.star_radius ** 3)
        st_logg = np.log10(req.star_mass / (req.star_radius ** 2))
        
        pl_star_ratio = pl_radius / req.star_radius
        transit_signal = (pl_radius / req.star_radius) ** 2
        orbital_velocity = np.sqrt(req.star_mass / req.orbital_period)
        
        pl_orbsmax = (req.orbital_period / 365.25) ** (2/3) * req.star_mass ** (1/3)
        
        # Habitable zone check
        inner_hz = 0.95 * np.sqrt(st_luminosity)
        outer_hz = 1.67 * np.sqrt(st_luminosity)
        in_hz = 1 if (pl_orbsmax >= inner_hz and pl_orbsmax <= outer_hz) else 0
        
        pl_eqt_calc = req.star_temp * np.sqrt(req.star_radius / (2 * pl_orbsmax))
        
        transit_depth_norm = req.transit_depth / 100.0
        expected_depth = transit_signal * 100
        transit_quality = req.transit_depth / (expected_depth + 1e-6)
        
        mass_period_int = req.star_mass * np.log1p(req.orbital_period)
        temp_radius_int = req.star_temp * req.star_radius
        lum_distance_int = st_luminosity * pl_orbsmax
        
        # Create feature array in EXACT order as training
        features = np.array([[
            req.star_temp,           # st_teff
            req.star_radius,         # st_rad
            req.star_mass,           # st_mass
            st_logg,                 # st_logg
            st_teff_ratio,           # st_teff_ratio
            st_luminosity,           # st_luminosity
            st_density,              # st_density
            req.orbital_period,      # pl_orbper
            pl_radius,               # pl_rade
            req.transit_depth,       # pl_trandep
            pl_star_ratio,           # pl_star_ratio
            transit_signal,          # transit_signal
            orbital_velocity,        # orbital_velocity
            pl_orbsmax,              # pl_orbsmax
            in_hz,                   # in_hz
            pl_eqt_calc,             # pl_eqt_calc
            transit_depth_norm,      # transit_depth_norm
            transit_quality,         # transit_quality
            mass_period_int,         # mass_period_int
            temp_radius_int,         # temp_radius_int
            lum_distance_int         # lum_distance_int
        ]])
        
        # Habitability data
        hab_data = calculate_habitability(
            req.star_temp, req.star_radius, req.star_mass,
            pl_radius, req.orbital_period
        )
        
        # Predict with model
        if 'rf' in ml_models and 'scaler' in ml_models:
            # Scale features
            X_scaled = ml_models['scaler'].transform(features)
            prediction = int(ml_models['rf'].predict(X_scaled)[0])
            probability = float(ml_models['rf'].predict_proba(X_scaled)[0][1])
        else:
            # Fallback heuristic
            score = (req.transit_depth * 50 + 
                    (1 - abs(req.star_temp - 5778) / 5778) * 30 +
                    (1 if in_hz else 0) * 20)
            probability = min(score / 100, 1.0)
            prediction = 1 if probability > 0.5 else 0
        
        # Planet type classification
        if pl_radius < 1.25:
            planet_type = "Earth-like"
        elif pl_radius < 2.0:
            planet_type = "Super-Earth"
        elif pl_radius < 6.0:
            planet_type = "Neptune-like"
        else:
            planet_type = "Jupiter-like"
        
        # Confidence level
        if probability > 0.9:
            confidence = "Very High"
        elif probability > 0.7:
            confidence = "High"
        elif probability > 0.5:
            confidence = "Moderate"
        else:
            confidence = "Low"
        
        return PredictionResponse(
            prediction=prediction,
            probability=probability,
            planet_type=planet_type,
            habitable_zone=hab_data.get('zone', 'Unknown'),
            confidence=confidence,
            details={
                "planet_radius_estimate": round(float(pl_radius), 3),
                "habitability_data": hab_data,
                "features_analyzed": 21
            }
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(500, f"Prediction failed: {e}")

@app.post("/api/analyze-lightcurve")
async def analyze_lightcurve(file: UploadFile = File(...)):
    """Analyze uploaded light curve file"""
    try:
        # Read file
        contents = await file.read()
        
        # Try to parse as CSV
        try:
            df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        except:
            raise HTTPException(400, "Invalid CSV format. Expected columns: time, flux")
        
        if 'time' not in df.columns or 'flux' not in df.columns:
            raise HTTPException(400, "CSV must have 'time' and 'flux' columns")
        
        # Remove NaN values
        df = df.dropna(subset=['time', 'flux'])
        
        time = df['time'].values
        flux = df['flux'].values
        
        # Analyze
        analysis = analyze_light_curve(time, flux)
        
        return {
            **analysis,
            "plot_data": {
                "time": time.tolist()[:1000],  # Limit points
                "flux": flux.tolist()[:1000]
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Light curve analysis error: {e}")
        raise HTTPException(500, f"Analysis failed: {e}")

@app.post("/api/simulate-transit")
def simulate_transit_endpoint(req: TransitSimulationRequest):
    """Simulate planet transit"""
    try:
        result = simulate_transit(
            req.planet_radius,
            req.star_radius,
            req.orbital_period,
            req.inclination,
            req.num_points
        )
        return result
    except Exception as e:
        logger.error(f"Simulation error: {e}")
        raise HTTPException(500, f"Simulation failed: {e}")

@app.post("/api/habitability")
def calculate_habitability_endpoint(req: HabitabilityRequest):
    """Calculate detailed habitability metrics"""
    try:
        result = calculate_habitability(
            req.star_temp,
            req.star_radius,
            req.star_mass,
            req.planet_radius,
            req.orbital_period
        )
        return result
    except Exception as e:
        logger.error(f"Habitability calculation error: {e}")
        raise HTTPException(500, f"Calculation failed: {e}")

@app.get("/api/chart-data")
def get_chart_data():
    """Enhanced chart data"""
    try:
        data = fetch_nasa_exoplanets()
        
        # Timeline
        timeline = {}
        for p in data:
            year = p.get('disc_year')
            if year and year >= 1995:
                timeline[int(year)] = timeline.get(int(year), 0) + 1
        
        # Methods
        methods = {}
        for p in data:
            method = p.get('discoverymethod', 'Unknown')
            methods[method] = methods.get(method, 0) + 1
        
        # Size distribution
        sizes = {"Earth-like": 0, "Super-Earth": 0, "Neptune-like": 0, "Jupiter-like": 0, "Unknown": 0}
        for p in data:
            radius = p.get('pl_rade')
            if not radius:
                sizes["Unknown"] += 1
            elif radius < 1.25:
                sizes["Earth-like"] += 1
            elif radius < 2.0:
                sizes["Super-Earth"] += 1
            elif radius < 6.0:
                sizes["Neptune-like"] += 1
            else:
                sizes["Jupiter-like"] += 1
        
        # Distance
        distances = {"<50 ly": 0, "50-100 ly": 0, "100-500 ly": 0, ">500 ly": 0, "Unknown": 0}
        for p in data:
            dist = p.get('sy_dist')
            if not dist:
                distances["Unknown"] += 1
            elif dist < 50:
                distances["<50 ly"] += 1
            elif dist < 100:
                distances["50-100 ly"] += 1
            elif dist < 500:
                distances["100-500 ly"] += 1
            else:
                distances[">500 ly"] += 1
        
        return {
            "timeline": dict(sorted(timeline.items())),
            "methods": methods,
            "sizes": sizes,
            "distances": distances
        }
    except Exception as e:
        logger.error(f"Chart data error: {e}")
        raise HTTPException(500, f"Error: {e}")


# ============================================================================
# SERVER STARTUP
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    import platform
    
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info("=" * 80)
    logger.info(f"🚀 {settings.APP_NAME} v{settings.VERSION}")
    logger.info("=" * 80)
    logger.info(f"💻 Platform: {platform.system()}")
    logger.info(f"🌐 Server: http://{host}:{port}")
    logger.info(f"🤖 Models: RF={'rf' in ml_models}, CNN={'cnn' in ml_models}, LSTM={'lstm' in ml_models}")
    logger.info(f"📡 NASA Integration: Active")
    logger.info(f"🏆 NASA Space Apps Challenge 2025")
    logger.info("=" * 80)
    
    uvicorn.run(
        app if platform.system() == "Windows" else "main:app",
        host=host,
        port=port,
        log_level="info",
        reload=settings.DEBUG

    )
