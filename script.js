/**
 * Stellar Scout V2 - Frontend JavaScript
 * NASA Space Apps Challenge 2025
 * Advanced AI-powered exoplanet detection system
 */

// ============================================================================
// CONFIGURATION
// ============================================================================

const API_BASE_URL = 'https://stellar-scout2-1.onrender.com';

let currentPage = 1;
let allExoplanets = [];
let filteredExoplanets = [];

// Initialize charts object
let charts = {
    lightcurve: null,
    simulation: null,
    timeline: null,
    methods: null,
    size: null,
    distance: null
};

// ============================================================================
// INITIALIZATION
// ============================================================================

document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 Stellar Scout V2 Initializing...');
    
    // Check dependencies
    checkDependencies();
    
    // Initialize components
    initializeNavbar();
    loadInitialData();
    initializeCharts();
    setupFileUpload();
    setupSimulatorControls();
    
    console.log('✅ Initialization complete!');
});

function checkDependencies() {
    if (typeof Chart === 'undefined') {
        console.error('❌ Chart.js NOT loaded!');
    } else {
        console.log('✅ Chart.js loaded');
    }
    
    if (typeof THREE === 'undefined') {
        console.warn('⚠️ Three.js NOT loaded!');
    } else {
        console.log('✅ Three.js loaded');
    }
}

// ============================================================================
// NAVIGATION
// ============================================================================

function initializeNavbar() {
    const navbar = document.querySelector('.navbar');
    const menuToggle = document.getElementById('menuToggle');
    const navLinks = document.querySelector('.nav-links');
    
    // Scroll effect
    window.addEventListener('scroll', function() {
        if (window.scrollY > 50) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }
    });
    
    // Mobile menu
    menuToggle.addEventListener('click', function() {
        menuToggle.classList.toggle('active');
        navLinks.classList.toggle('active');
    });
    
    // Close menu on link click
    document.querySelectorAll('.nav-links a').forEach(link => {
        link.addEventListener('click', function() {
            menuToggle.classList.remove('active');
            navLinks.classList.remove('active');
        });
    });
}

function scrollToSection(sectionId) {
    const section = document.getElementById(sectionId);
    section.scrollIntoView({ behavior: 'smooth' });
}

// ============================================================================
// DATA LOADING
// ============================================================================

async function loadInitialData() {
    console.log('📡 Loading initial data...');
    
    try {
        await loadStats();
        await loadExoplanets();
        await loadChartData();
    } catch (error) {
        console.error('Error loading data:', error);
    }
}

async function loadStats() {
    try {
        const response = await fetch(`${API_BASE_URL}/stats`);
        const data = await response.json();
        
        animateCounter('totalExoplanets', data.total_exoplanets);
        animateCounter('habitableExoplanets', data.potentially_habitable);
        animateCounter('recentDiscoveries', data.recent_discoveries);
        
        console.log('✅ Stats loaded');
    } catch (error) {
        console.error('Stats error:', error);
        document.getElementById('totalExoplanets').textContent = '5500+';
        document.getElementById('habitableExoplanets').textContent = '60+';
        document.getElementById('recentDiscoveries').textContent = '150+';
    }
}

function animateCounter(elementId, targetValue) {
    const element = document.getElementById(elementId);
    const duration = 2000;
    const start = 0;
    const increment = targetValue / (duration / 16);
    let current = start;
    
    const timer = setInterval(() => {
        current += increment;
        if (current >= targetValue) {
            element.textContent = targetValue.toLocaleString();
            clearInterval(timer);
        } else {
            element.textContent = Math.floor(current).toLocaleString();
        }
    }, 16);
}

// ============================================================================
// EXOPLANET EXPLORER
// ============================================================================

async function loadExoplanets(page = 1) {
    const grid = document.getElementById('exoplanetGrid');
    
    if (page === 1) {
        grid.innerHTML = '<div class="loading-spinner"><div class="spinner"></div><p>Loading exoplanets from NASA...</p></div>';
    }
    
    try {
        const searchTerm = document.getElementById('searchInput')?.value || '';
        const method = document.getElementById('methodFilter')?.value || '';
        const minRadius = document.getElementById('minRadiusFilter')?.value || '';
        const maxRadius = document.getElementById('maxRadiusFilter')?.value || '';
        
        let url = `${API_BASE_URL}/exoplanets?page=${page}&limit=12`;
        if (searchTerm) url += `&search=${encodeURIComponent(searchTerm)}`;
        if (method) url += `&method=${encodeURIComponent(method)}`;
        if (minRadius) url += `&min_radius=${minRadius}`;
        if (maxRadius) url += `&max_radius=${maxRadius}`;
        
        const response = await fetch(url);
        const data = await response.json();
        
        if (page === 1) {
            allExoplanets = data.exoplanets;
            displayExoplanets(allExoplanets);
        } else {
            allExoplanets = [...allExoplanets, ...data.exoplanets];
            appendExoplanets(data.exoplanets);
        }
        
        currentPage = page;
        
        const loadMoreBtn = document.getElementById('loadMoreBtn');
        if (data.has_more) {
            loadMoreBtn.style.display = 'block';
        } else {
            loadMoreBtn.style.display = 'none';
        }
        
        console.log(`✅ Loaded ${data.exoplanets.length} exoplanets (page ${page})`);
    } catch (error) {
        console.error('Error loading exoplanets:', error);
        grid.innerHTML = '<p style="text-align: center; color: var(--text-secondary);">Failed to load exoplanets. Please try again.</p>';
    }
}

function displayExoplanets(exoplanets) {
    const grid = document.getElementById('exoplanetGrid');
    
    if (exoplanets.length === 0) {
        grid.innerHTML = '<p style="text-align: center; grid-column: 1 / -1; color: var(--text-secondary);">No exoplanets found matching your criteria.</p>';
        return;
    }
    
    grid.innerHTML = '';
    exoplanets.forEach(planet => {
        grid.appendChild(createPlanetCard(planet));
    });
}

function appendExoplanets(exoplanets) {
    const grid = document.getElementById('exoplanetGrid');
    exoplanets.forEach(planet => {
        grid.appendChild(createPlanetCard(planet));
    });
}

function createPlanetCard(planet) {
    const card = document.createElement('div');
    card.className = 'planet-card';
    card.onclick = () => showPlanetDetails(planet);
    
    const planetType = getPlanetType(planet.pl_rade);
    
    card.innerHTML = `
        <h3 class="planet-name">${planet.pl_name || 'Unknown'}</h3>
        <div class="planet-info">
            <div class="info-row">
                <span class="info-label">Host Star</span>
                <span class="info-value">${planet.hostname || 'Unknown'}</span>
            </div>
            <div class="info-row">
                <span class="info-label">Discovery Method</span>
                <span class="info-value">${planet.discoverymethod || 'Unknown'}</span>
            </div>
            <div class="info-row">
                <span class="info-label">Discovery Year</span>
                <span class="info-value">${planet.disc_year || 'Unknown'}</span>
            </div>
            <div class="info-row">
                <span class="info-label">Distance</span>
                <span class="info-value">${planet.sy_dist ? planet.sy_dist.toFixed(2) + ' ly' : 'Unknown'}</span>
            </div>
            <div class="info-row">
                <span class="info-label">Planet Radius</span>
                <span class="info-value">${planet.pl_rade ? planet.pl_rade.toFixed(2) + ' R⊕' : 'Unknown'}</span>
            </div>
        </div>
        <span class="planet-badge">${planetType}</span>
    `;
    
    return card;
}

function getPlanetType(radius) {
    if (!radius) return 'Unknown Type';
    
    if (radius < 1.25) return 'Earth-like';
    if (radius < 2.0) return 'Super-Earth';
    if (radius < 6.0) return 'Neptune-like';
    return 'Jupiter-like';
}

function searchExoplanets() {
    currentPage = 1;
    loadExoplanets(1);
}

function applyFilters() {
    currentPage = 1;
    loadExoplanets(1);
}

function resetFilters() {
    document.getElementById('searchInput').value = '';
    document.getElementById('methodFilter').value = '';
    document.getElementById('minRadiusFilter').value = '';
    document.getElementById('maxRadiusFilter').value = '';
    currentPage = 1;
    loadExoplanets(1);
}

function loadMoreExoplanets() {
    loadExoplanets(currentPage + 1);
}

// ============================================================================
// PLANET DETAILS MODAL
// ============================================================================

function showPlanetDetails(planet) {
    const modal = document.getElementById('planetModal');
    const modalBody = document.getElementById('modalBody');
    
    modalBody.innerHTML = `
        <h2 class="modal-planet-name">${planet.pl_name || 'Unknown'}</h2>
        <p class="modal-planet-host">Orbits ${planet.hostname || 'Unknown Star'}</p>
        
        <div class="modal-info-grid">
            <div class="modal-info-item">
                <div class="modal-info-label">Discovery Method</div>
                <div class="modal-info-value">${planet.discoverymethod || 'Unknown'}</div>
            </div>
            <div class="modal-info-item">
                <div class="modal-info-label">Discovery Year</div>
                <div class="modal-info-value">${planet.disc_year || 'Unknown'}</div>
            </div>
            <div class="modal-info-item">
                <div class="modal-info-label">Planet Radius</div>
                <div class="modal-info-value">${planet.pl_rade ? planet.pl_rade.toFixed(2) + ' R⊕' : 'Unknown'}</div>
            </div>
            <div class="modal-info-item">
                <div class="modal-info-label">Planet Mass</div>
                <div class="modal-info-value">${planet.pl_bmasse ? planet.pl_bmasse.toFixed(2) + ' M⊕' : 'Unknown'}</div>
            </div>
            <div class="modal-info-item">
                <div class="modal-info-label">Orbital Period</div>
                <div class="modal-info-value">${planet.pl_orbper ? planet.pl_orbper.toFixed(2) + ' days' : 'Unknown'}</div>
            </div>
            <div class="modal-info-item">
                <div class="modal-info-label">Distance from Earth</div>
                <div class="modal-info-value">${planet.sy_dist ? planet.sy_dist.toFixed(2) + ' ly' : 'Unknown'}</div>
            </div>
            <div class="modal-info-item">
                <div class="modal-info-label">Star Temperature</div>
                <div class="modal-info-value">${planet.st_teff ? planet.st_teff + ' K' : 'Unknown'}</div>
            </div>
            <div class="modal-info-item">
                <div class="modal-info-label">Star Radius</div>
                <div class="modal-info-value">${planet.st_rad ? planet.st_rad.toFixed(2) + ' R☉' : 'Unknown'}</div>
            </div>
            <div class="modal-info-item">
                <div class="modal-info-label">Equilibrium Temp</div>
                <div class="modal-info-value">${planet.pl_eqt ? planet.pl_eqt.toFixed(0) + ' K' : 'Unknown'}</div>
            </div>
            <div class="modal-info-item">
                <div class="modal-info-label">Discovery Facility</div>
                <div class="modal-info-value">${planet.disc_facility || 'Unknown'}</div>
            </div>
        </div>
    `;
    
    modal.classList.add('active');
}

function closeModal() {
    const modal = document.getElementById('planetModal');
    modal.classList.remove('active');
}

window.onclick = function(event) {
    const modal = document.getElementById('planetModal');
    if (event.target === modal) {
        closeModal();
    }
}

// ============================================================================
// AI PREDICTION
// ============================================================================

async function predictExoplanet(event) {
    event.preventDefault();
    
    const resultsPanel = document.getElementById('resultsPanel');
    resultsPanel.innerHTML = '<div class="loading-spinner"><div class="spinner"></div><p>AI is analyzing...</p></div>';
    
    const formData = {
        star_temp: parseFloat(document.getElementById('starTemp').value),
        star_radius: parseFloat(document.getElementById('starRadius').value),
        star_mass: parseFloat(document.getElementById('starMass').value),
        orbital_period: parseFloat(document.getElementById('orbitalPeriod').value),
        transit_depth: parseFloat(document.getElementById('transitDepth').value)
    };
    
    try {
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(formData)
        });
        
        const result = await response.json();
        displayPredictionResult(result);
        console.log('✅ Prediction complete:', result);
    } catch (error) {
        console.error('Prediction error:', error);
        resultsPanel.innerHTML = '<p style="text-align: center; color: var(--danger-color);">Prediction failed. Please check your inputs and try again.</p>';
    }
}

function displayPredictionResult(result) {
    const resultsPanel = document.getElementById('resultsPanel');
    
    const hasExoplanet = result.prediction === 1;
    const confidence = (result.probability * 100).toFixed(1);
    
    const statusIcon = hasExoplanet ? '✅' : '❌';
    const statusText = hasExoplanet ? 'Exoplanet Detected!' : 'No Exoplanet Detected';
    const statusColor = hasExoplanet ? 'var(--success-color)' : 'var(--danger-color)';
    
    resultsPanel.innerHTML = `
        <h3 style="display: flex; align-items: center; gap: 10px; margin-bottom: 30px;">
            📊 AI Analysis Results
        </h3>
        <div class="prediction-result" style="animation: fadeInUp 0.5s;">
            <div class="result-header" style="text-align: center; margin-bottom: 30px;">
                <div class="result-status" style="font-size: 72px; margin-bottom: 15px;">${statusIcon}</div>
                <div class="result-title" style="font-size: 28px; font-weight: 700; color: ${statusColor}; margin-bottom: 10px;">
                    ${statusText}
                </div>
                <p style="color: var(--text-secondary); font-size: 16px;">
                    ${hasExoplanet ? 'Our AI detected strong evidence of an exoplanet' : 'No significant exoplanet signals detected'}
                </p>
            </div>
            
            <div class="confidence-bar" style="width: 100%; height: 40px; background: rgba(255,255,255,0.05); border-radius: 20px; overflow: hidden; margin: 25px 0;">
                <div class="confidence-fill" style="width: ${confidence}%; height: 100%; background: linear-gradient(90deg, var(--primary-color), var(--accent-color)); display: flex; align-items: center; justify-content: center; color: white; font-weight: 700; font-size: 18px; transition: width 1s;">
                    ${confidence}% Confidence
                </div>
            </div>
            
            <div class="result-details" style="margin-top: 30px;">
                <div class="detail-item" style="padding: 15px; background: rgba(255,255,255,0.03); border-radius: 12px; margin-bottom: 12px; display: flex; justify-content: space-between;">
                    <span style="color: var(--text-secondary);">Prediction</span>
                    <span style="color: var(--text-primary); font-weight: 600;">${hasExoplanet ? 'Positive' : 'Negative'}</span>
                </div>
                <div class="detail-item" style="padding: 15px; background: rgba(255,255,255,0.03); border-radius: 12px; margin-bottom: 12px; display: flex; justify-content: space-between;">
                    <span style="color: var(--text-secondary);">Confidence Level</span>
                    <span style="color: var(--text-primary); font-weight: 600;">${result.confidence || 'Unknown'}</span>
                </div>
                <div class="detail-item" style="padding: 15px; background: rgba(255,255,255,0.03); border-radius: 12px; margin-bottom: 12px; display: flex; justify-content: space-between;">
                    <span style="color: var(--text-secondary);">Planet Type</span>
                    <span style="color: var(--text-primary); font-weight: 600;">${result.planet_type || 'Unknown'}</span>
                </div>
                <div class="detail-item" style="padding: 15px; background: rgba(255,255,255,0.03); border-radius: 12px; margin-bottom: 12px; display: flex; justify-content: space-between;">
                    <span style="color: var(--text-secondary);">Habitable Zone</span>
                    <span style="color: var(--text-primary); font-weight: 600;">${result.habitable_zone || 'Unknown'}</span>
                </div>
                ${result.details && result.details.habitability_data ? `
                <div class="detail-item" style="padding: 15px; background: rgba(255,255,255,0.03); border-radius: 12px; margin-bottom: 12px; display: flex; justify-content: space-between;">
                    <span style="color: var(--text-secondary);">Equilibrium Temperature</span>
                    <span style="color: var(--text-primary); font-weight: 600;">${result.details.habitability_data.equilibrium_temp || 'N/A'} K</span>
                </div>
                <div class="detail-item" style="padding: 15px; background: rgba(255,255,255,0.03); border-radius: 12px; display: flex; justify-content: space-between;">
                    <span style="color: var(--text-secondary);">Earth Similarity Index</span>
                    <span style="color: var(--text-primary); font-weight: 600;">${result.details.habitability_data.earth_similarity_index || 'N/A'}</span>
                </div>
                ` : ''}
            </div>
        </div>
    `;
}

// ============================================================================
// LIGHT CURVE ANALYZER
// ============================================================================

function setupFileUpload() {
    const fileInput = document.getElementById('lightcurveFile');
    const uploadZone = document.getElementById('uploadZone');
    
    if (!fileInput || !uploadZone) return;
    
    // Drag and drop
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.style.borderColor = 'var(--primary-color)';
        uploadZone.style.background = 'rgba(102, 126, 234, 0.1)';
    });
    
    uploadZone.addEventListener('dragleave', () => {
        uploadZone.style.borderColor = 'var(--border-color)';
        uploadZone.style.background = '';
    });
    
    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.style.borderColor = 'var(--border-color)';
        uploadZone.style.background = '';
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileUpload(files[0]);
        }
    });
    
    // File input
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });
}

async function handleFileUpload(file) {
    console.log('📁 Uploading file:', file.name);
    
    const viz = document.getElementById('lightcurveViz');
    viz.querySelector('h3').innerHTML = '📈 Analyzing Light Curve...';
    viz.querySelector('.viz-placeholder')?.remove();
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch(`${API_BASE_URL}/analyze-lightcurve`, {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        displayLightCurveResult(result);
        console.log('✅ Light curve analyzed:', result);
    } catch (error) {
        console.error('Light curve analysis error:', error);
        alert('Analysis failed: ' + error.message);
    }
}

function displayLightCurveResult(result) {
    const viz = document.getElementById('lightcurveViz');
    viz.querySelector('h3').innerHTML = '📈 Light Curve Visualization';
    
    // Create chart only if we have valid data
    const canvas = document.getElementById('lightcurveChart');
    if (canvas && result.plot_data && result.plot_data.time && result.plot_data.flux) {
        const ctx = canvas.getContext('2d');
        
        // Destroy previous chart if exists
        if (charts.lightcurve) {
            charts.lightcurve.destroy();
        }
        
        charts.lightcurve = new Chart(ctx, {
            type: 'line',
            data: {
                labels: result.plot_data.time,
                datasets: [{
                    label: 'Normalized Flux',
                    data: result.plot_data.flux,
                    borderColor: 'var(--primary-color)',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        titleColor: '#fff',
                        bodyColor: '#fff'
                    }
                },
                scales: {
                    x: {
                        title: { display: true, text: 'Time', color: 'var(--text-secondary)' },
                        ticks: { color: 'var(--text-secondary)' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    },
                    y: {
                        title: { display: true, text: 'Flux', color: 'var(--text-secondary)' },
                        ticks: { color: 'var(--text-secondary)' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    }
                }
            }
        });
    }
    
    // Display results with proper null checks
    const resultsDiv = document.getElementById('lightcurveResults');
    resultsDiv.innerHTML = `
        <div style="margin-top: 30px; padding: 30px; background: var(--card-bg); border-radius: 20px; border: 1px solid var(--border-color);">
            <h4 style="font-size: 24px; margin-bottom: 20px; color: var(--accent-color);">Analysis Results</h4>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                <div style="padding: 15px; background: rgba(255,255,255,0.03); border-radius: 12px;">
                    <div style="color: var(--text-muted); font-size: 14px; margin-bottom: 5px;">Transit Detected</div>
                    <div style="font-size: 24px; font-weight: 700; color: ${result.has_transit ? 'var(--success-color)' : 'var(--danger-color)'};">
                        ${result.has_transit ? 'YES ✔' : 'NO ✗'}
                    </div>
                </div>
                <div style="padding: 15px; background: rgba(255,255,255,0.03); border-radius: 12px;">
                    <div style="color: var(--text-muted); font-size: 14px; margin-bottom: 5px;">Confidence</div>
                    <div style="font-size: 24px; font-weight: 700; color: var(--primary-color);">
                        ${result.confidence !== undefined && result.confidence !== null ? (result.confidence * 100).toFixed(1) : '0.0'}%
                    </div>
                </div>
                <div style="padding: 15px; background: rgba(255,255,255,0.03); border-radius: 12px;">
                    <div style="color: var(--text-muted); font-size: 14px; margin-bottom: 5px;">Transit Depth</div>
                    <div style="font-size: 24px; font-weight: 700; color: var(--text-primary);">
                        ${result.transit_depth !== undefined && result.transit_depth !== null ? (result.transit_depth * 100).toFixed(3) : '0.000'}%
                    </div>
                </div>
                <div style="padding: 15px; background: rgba(255,255,255,0.03); border-radius: 12px;">
                    <div style="color: var(--text-muted); font-size: 14px; margin-bottom: 5px;">Orbital Period</div>
                    <div style="font-size: 24px; font-weight: 700; color: var(--text-primary);">
                        ${result.orbital_period !== undefined && result.orbital_period !== null ? result.orbital_period.toFixed(2) : '0.00'} days
                    </div>
                </div>
                <div style="padding: 15px; background: rgba(255,255,255,0.03); border-radius: 12px;">
                    <div style="color: var(--text-muted); font-size: 14px; margin-bottom: 5px;">Planet Radius Est.</div>
                    <div style="font-size: 24px; font-weight: 700; color: var(--text-primary);">
                        ${result.planet_radius_estimate !== undefined && result.planet_radius_estimate !== null ? result.planet_radius_estimate.toFixed(2) : '0.00'} R⊕
                    </div>
                </div>
                <div style="padding: 15px; background: rgba(255,255,255,0.03); border-radius: 12px;">
                    <div style="color: var(--text-muted); font-size: 14px; margin-bottom: 5px;">Planet Type</div>
                    <div style="font-size: 24px; font-weight: 700; color: var(--text-primary);">
                        ${result.planet_radius_estimate !== undefined && result.planet_radius_estimate !== null ? getPlanetType(result.planet_radius_estimate) : 'Unknown'}
                    </div>
                </div>
            </div>
        </div>
    `;
}

// ============================================================================
// TRANSIT SIMULATOR
// ============================================================================

function setupSimulatorControls() {
    const controls = ['simPlanetRadius', 'simStarRadius', 'simOrbitalPeriod', 'simInclination'];
    
    controls.forEach(id => {
        const slider = document.getElementById(id);
        const display = document.getElementById(id + 'Value');
        
        if (slider && display) {
            slider.addEventListener('input', (e) => {
                display.textContent = e.target.value;
            });
        }
    });
}

async function runSimulation() {
    console.log('🎮 Running transit simulation...');
    
    const planet_radius = parseFloat(document.getElementById('simPlanetRadius').value);
    const star_radius = parseFloat(document.getElementById('simStarRadius').value);
    const orbital_period = parseFloat(document.getElementById('simOrbitalPeriod').value);
    const inclination = parseFloat(document.getElementById('simInclination').value);
    
    const requestData = {
        planet_radius,
        star_radius,
        orbital_period,
        inclination,
        num_points: 1000
    };
    
    try {
        const response = await fetch(`${API_BASE_URL}/simulate-transit`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestData)
        });
        
        const result = await response.json();
        displaySimulationResult(result);
        console.log('✅ Simulation complete');
    } catch (error) {
        console.error('Simulation error:', error);
        alert('Simulation failed: ' + error.message);
    }
}

function displaySimulationResult(result) {
    const canvas = document.getElementById('simulationChart');
    const ctx = canvas.getContext('2d');
    
    // Clear previous chart
    if (charts.simulation) {
        charts.simulation.destroy();
    }
    
    charts.simulation = new Chart(ctx, {
        type: 'line',
        data: {
            labels: result.time,
            datasets: [{
                label: 'Normalized Flux',
                data: result.flux,
                borderColor: 'var(--success-color)',
                backgroundColor: 'rgba(67, 233, 123, 0.1)',
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: '#fff',
                    bodyColor: '#fff'
                }
            },
            scales: {
                x: {
                    title: { display: true, text: 'Time (days)', color: 'var(--text-secondary)' },
                    ticks: { color: 'var(--text-secondary)' },
                    grid: { color: 'rgba(255, 255, 255, 0.1)' }
                },
                y: {
                    title: { display: true, text: 'Normalized Flux', color: 'var(--text-secondary)' },
                    ticks: { color: 'var(--text-secondary)' },
                    grid: { color: 'rgba(255, 255, 255, 0.1)' }
                }
            }
        }
    });
    
    // Display stats
    const statsDiv = document.getElementById('simStats');
    statsDiv.innerHTML = `
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; margin-top: 20px;">
            <div style="padding: 20px; background: var(--card-bg); border-radius: 16px; border: 1px solid var(--border-color);">
                <div style="color: var(--text-muted); font-size: 14px; margin-bottom: 8px;">Transit Depth</div>
                <div style="font-size: 28px; font-weight: 700; color: var(--success-color);">
                    ${(result.transit_depth * 100).toFixed(3)}%
                </div>
            </div>
            <div style="padding: 20px; background: var(--card-bg); border-radius: 16px; border: 1px solid var(--border-color);">
                <div style="color: var(--text-muted); font-size: 14px; margin-bottom: 8px;">Transit Duration</div>
                <div style="font-size: 28px; font-weight: 700; color: var(--success-color);">
                    ${result.duration_hours.toFixed(2)} hrs
                </div>
            </div>
        </div>
    `;
    
    const infoDiv = document.getElementById('simulationInfo');
    infoDiv.innerHTML = `
        <div style="padding: 20px; background: rgba(67, 233, 123, 0.1); border-radius: 16px; border: 1px solid var(--success-color);">
            <h4 style="color: var(--success-color); margin-bottom: 10px;">✅ Simulation Complete</h4>
            <p style="color: var(--text-secondary); font-size: 14px;">
                The simulated transit shows a ${(result.transit_depth * 100).toFixed(3)}% dip in stellar brightness 
                lasting approximately ${result.duration_hours.toFixed(2)} hours.
            </p>
        </div>
    `;
}

// ============================================================================
// CHARTS & VISUALIZATIONS
// ============================================================================

function initializeCharts() {
    console.log('📊 Initializing charts...');
}

async function loadChartData() {
    try {
        const response = await fetch(`${API_BASE_URL}/chart-data`);
        const data = await response.json();
        
        createTimelineChart(data.timeline);
        createMethodsChart(data.methods);
        createSizeChart(data.sizes);
        createDistanceChart(data.distances);
        
        console.log('✅ Charts loaded');
    } catch (error) {
        console.error('Chart data error:', error);
        createDemoCharts();
    }
}

function createTimelineChart(data) {
    const ctx = document.getElementById('timelineChart');
    if (!ctx || typeof Chart === 'undefined') return;
    
    const years = Object.keys(data);
    const counts = Object.values(data);
    
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: years,
            datasets: [{
                label: 'Exoplanets Discovered',
                data: counts,
                borderColor: '#667eea',
                backgroundColor: 'rgba(102, 126, 234, 0.2)',
                tension: 0.4,
                fill: true,
                pointBackgroundColor: '#667eea',
                pointBorderColor: '#fff',
                pointBorderWidth: 2,
                pointRadius: 4,
                pointHoverRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: '#fff',
                    bodyColor: '#fff',
                    borderColor: '#667eea',
                    borderWidth: 1
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    ticks: { color: '#b8c5d6' }
                },
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#b8c5d6', maxRotation: 45, minRotation: 45 }
                }
            }
        }
    });
}

function createMethodsChart(data) {
    const ctx = document.getElementById('methodsChart');
    if (!ctx || typeof Chart === 'undefined') return;
    
    const methods = Object.keys(data);
    const counts = Object.values(data);
    
    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: methods,
            datasets: [{
                data: counts,
                backgroundColor: ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#43e97b'],
                borderColor: '#0a0e27',
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { color: '#b8c5d6', padding: 15, font: { size: 12 } }
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: '#fff',
                    bodyColor: '#fff',
                    borderColor: '#667eea',
                    borderWidth: 1
                }
            }
        }
    });
}

function createSizeChart(data) {
    const ctx = document.getElementById('sizeChart');
    if (!ctx || typeof Chart === 'undefined') return;
    
    const sizes = Object.keys(data);
    const counts = Object.values(data);
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: sizes,
            datasets: [{
                label: 'Number of Planets',
                data: counts,
                backgroundColor: ['rgba(102, 126, 234, 0.8)', 'rgba(118, 75, 162, 0.8)', 'rgba(240, 147, 251, 0.8)', 'rgba(79, 172, 254, 0.8)', 'rgba(67, 233, 123, 0.8)'],
                borderColor: ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#43e97b'],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: '#fff',
                    bodyColor: '#fff',
                    borderColor: '#667eea',
                    borderWidth: 1
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    ticks: { color: '#b8c5d6' }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: '#b8c5d6' }
                }
            }
        }
    });
}

function createDistanceChart(data) {
    const ctx = document.getElementById('distanceChart');
    if (!ctx || typeof Chart === 'undefined') return;
    
    const distances = Object.keys(data);
    const counts = Object.values(data);
    
    new Chart(ctx, {
        type: 'polarArea',
        data: {
            labels: distances,
            datasets: [{
                data: counts,
                backgroundColor: ['rgba(102, 126, 234, 0.6)', 'rgba(118, 75, 162, 0.6)', 'rgba(240, 147, 251, 0.6)', 'rgba(79, 172, 254, 0.6)', 'rgba(67, 233, 123, 0.6)'],
                borderColor: ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#43e97b'],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { color: '#b8c5d6', padding: 15, font: { size: 12 } }
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: '#fff',
                    bodyColor: '#fff',
                    borderColor: '#667eea',
                    borderWidth: 1
                }
            },
            scales: {
                r: {
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    ticks: { color: '#b8c5d6', backdropColor: 'transparent' }
                }
            }
        }
    });
}

function createDemoCharts() {
    console.log('📊 Creating demo charts...');
    
    const timelineData = {
        '2015': 142, '2016': 1284, '2017': 219, '2018': 289,
        '2019': 143, '2020': 89, '2021': 134, '2022': 153,
        '2023': 197, '2024': 178
    };
    
    const methodsData = {
        'Transit': 3869, 'Radial Velocity': 1056,
        'Imaging': 72, 'Microlensing': 187, 'Other': 58
    };
    
    const sizeData = {
        'Earth-like': 892, 'Super-Earth': 1653,
        'Neptune-like': 1876, 'Jupiter-like': 1421, 'Unknown': 400
    };
    
    const distanceData = {
        '<50 ly': 234, '50-100 ly': 456,
        '100-500 ly': 2134, '>500 ly': 1876, 'Unknown': 542
    };
    
    createTimelineChart(timelineData);
    createMethodsChart(methodsData);
    createSizeChart(sizeData);
    createDistanceChart(distanceData);
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Add debounced search
const debouncedSearch = debounce(searchExoplanets, 300);
document.addEventListener('DOMContentLoaded', function() {
    const searchInput = document.getElementById('searchInput');
    if (searchInput) {
        searchInput.addEventListener('input', debouncedSearch);
    }
});

// ============================================================================
// ERROR HANDLING
// ============================================================================

window.addEventListener('error', (e) => {
    console.error('Global error:', e.error);
});

window.addEventListener('unhandledrejection', (e) => {
    console.error('Unhandled promise rejection:', e.reason);
});

// ============================================================================
// CONSOLE BRANDING
// ============================================================================

console.log('%c🌟 Stellar Scout V2', 'font-size: 24px; font-weight: bold; color: #667eea;');
console.log('%cNASA Space Apps Challenge 2025', 'font-size: 14px; color: #b8c5d6;');
console.log('%cAdvanced AI-powered exoplanet detection system', 'font-size: 12px; color: #7a8ba3;');
console.log('%c━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━', 'color: #667eea;');

// ============================================================================
// EXPORT (for testing)
// ============================================================================

if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        loadExoplanets,
        predictExoplanet,
        runSimulation,
        loadChartData
    };

}
