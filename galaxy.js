/**
 * 3D Galaxy Background with Three.js
 * Creates a realistic animated galaxy with particles, nebula effects, and depth
 */

// Three.js Scene Setup
let scene, camera, renderer;
let galaxyParticles, nebulaParticles, stars;
let mouseX = 0, mouseY = 0;
let time = 0;

// Initialize 3D Galaxy
function initGalaxy() {
    const canvas = document.getElementById('galaxyCanvas');
    
    // Scene
    scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x000000, 0.0008);
    
    // Camera
    camera = new THREE.PerspectiveCamera(
        75,
        window.innerWidth / window.innerHeight,
        0.1,
        2000
    );
    camera.position.z = 500;
    
    // Renderer
    renderer = new THREE.WebGLRenderer({
        canvas: canvas,
        alpha: true,
        antialias: true
    });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    
    // Create galaxy components
    createGalaxySpiralArms();
    createNebulaCloud();
    createStarField();
    createDistantGalaxies();
    
    // Lighting
    const ambientLight = new THREE.AmbientLight(0x222244, 0.5);
    scene.add(ambientLight);
    
    const pointLight = new THREE.PointLight(0x4466ff, 1, 1000);
    pointLight.position.set(0, 0, 0);
    scene.add(pointLight);
    
    // Event listeners
    document.addEventListener('mousemove', onMouseMove, false);
    window.addEventListener('resize', onWindowResize, false);
    
    // Start animation
    animate();
    
    console.log('🌌 3D Galaxy Background Loaded');
}

// Create spiral galaxy arms with particles
function createGalaxySpiralArms() {
    const particleCount = 15000;
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(particleCount * 3);
    const colors = new Float32Array(particleCount * 3);
    const sizes = new Float32Array(particleCount);
    
    // Galaxy parameters
    const arms = 3;
    const armRotation = Math.PI * 2 / arms;
    
    for (let i = 0; i < particleCount; i++) {
        const i3 = i * 3;
        
        // Spiral arm calculation
        const armIndex = i % arms;
        const radius = Math.random() * 400 + 50;
        const spinAngle = radius * 0.01;
        const angle = armIndex * armRotation + spinAngle + Math.random() * 0.5;
        
        // Position with some randomness
        const x = Math.cos(angle) * radius + (Math.random() - 0.5) * 30;
        const y = (Math.random() - 0.5) * 20 * (1 - radius / 400);
        const z = Math.sin(angle) * radius + (Math.random() - 0.5) * 30;
        
        positions[i3] = x;
        positions[i3 + 1] = y;
        positions[i3 + 2] = z;
        
        // Color gradient from center (blue/purple) to edges (red/orange)
        const distanceRatio = radius / 400;
        
        if (distanceRatio < 0.3) {
            // Core: bright blue/white
            colors[i3] = 0.4 + Math.random() * 0.6;     // R
            colors[i3 + 1] = 0.6 + Math.random() * 0.4; // G
            colors[i3 + 2] = 1.0;                        // B
        } else if (distanceRatio < 0.7) {
            // Middle: purple/pink
            colors[i3] = 0.6 + Math.random() * 0.4;     // R
            colors[i3 + 1] = 0.3 + Math.random() * 0.3; // G
            colors[i3 + 2] = 0.8 + Math.random() * 0.2; // B
        } else {
            // Outer: red/orange
            colors[i3] = 0.8 + Math.random() * 0.2;     // R
            colors[i3 + 1] = 0.4 + Math.random() * 0.2; // G
            colors[i3 + 2] = 0.2 + Math.random() * 0.2; // B
        }
        
        // Size variation
        sizes[i] = Math.random() * 3 + 0.5;
    }
    
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));
    
    // Custom shader material for better looking particles
    const material = new THREE.ShaderMaterial({
        uniforms: {
            time: { value: 0 }
        },
        vertexShader: `
            attribute float size;
            attribute vec3 color;
            varying vec3 vColor;
            uniform float time;
            
            void main() {
                vColor = color;
                vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
                gl_PointSize = size * (300.0 / -mvPosition.z);
                gl_Position = projectionMatrix * mvPosition;
            }
        `,
        fragmentShader: `
            varying vec3 vColor;
            
            void main() {
                float dist = length(gl_PointCoord - vec2(0.5));
                if (dist > 0.5) discard;
                
                float alpha = 1.0 - (dist * 2.0);
                alpha = pow(alpha, 2.0);
                
                gl_FragColor = vec4(vColor, alpha);
            }
        `,
        transparent: true,
        vertexColors: true,
        blending: THREE.AdditiveBlending,
        depthWrite: false
    });
    
    galaxyParticles = new THREE.Points(geometry, material);
    scene.add(galaxyParticles);
}

// Create nebula cloud effect
function createNebulaCloud() {
    const particleCount = 3000;
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(particleCount * 3);
    const colors = new Float32Array(particleCount * 3);
    const sizes = new Float32Array(particleCount);
    
    for (let i = 0; i < particleCount; i++) {
        const i3 = i * 3;
        
        // Nebula distribution
        const theta = Math.random() * Math.PI * 2;
        const phi = Math.random() * Math.PI;
        const radius = Math.random() * 300 + 100;
        
        positions[i3] = radius * Math.sin(phi) * Math.cos(theta);
        positions[i3 + 1] = radius * Math.sin(phi) * Math.sin(theta) * 0.3;
        positions[i3 + 2] = radius * Math.cos(phi);
        
        // Nebula colors (pink, purple, blue)
        const colorType = Math.random();
        if (colorType < 0.33) {
            // Pink nebula
            colors[i3] = 1.0;
            colors[i3 + 1] = 0.2 + Math.random() * 0.3;
            colors[i3 + 2] = 0.6 + Math.random() * 0.4;
        } else if (colorType < 0.66) {
            // Purple nebula
            colors[i3] = 0.6 + Math.random() * 0.4;
            colors[i3 + 1] = 0.2 + Math.random() * 0.2;
            colors[i3 + 2] = 1.0;
        } else {
            // Blue nebula
            colors[i3] = 0.2 + Math.random() * 0.3;
            colors[i3 + 1] = 0.4 + Math.random() * 0.4;
            colors[i3 + 2] = 1.0;
        }
        
        sizes[i] = Math.random() * 20 + 5;
    }
    
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));
    
    const material = new THREE.PointsMaterial({
        size: 15,
        vertexColors: true,
        transparent: true,
        opacity: 0.15,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
        sizeAttenuation: true
    });
    
    nebulaParticles = new THREE.Points(geometry, material);
    scene.add(nebulaParticles);
}

// Create star field background
function createStarField() {
    const starCount = 5000;
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(starCount * 3);
    const colors = new Float32Array(starCount * 3);
    const sizes = new Float32Array(starCount);
    
    for (let i = 0; i < starCount; i++) {
        const i3 = i * 3;
        
        // Random distribution in sphere
        positions[i3] = (Math.random() - 0.5) * 1500;
        positions[i3 + 1] = (Math.random() - 0.5) * 1500;
        positions[i3 + 2] = (Math.random() - 0.5) * 1500;
        
        // Star colors (white, blue, yellow, red)
        const starType = Math.random();
        if (starType < 0.7) {
            // White stars
            colors[i3] = 1.0;
            colors[i3 + 1] = 1.0;
            colors[i3 + 2] = 1.0;
        } else if (starType < 0.85) {
            // Blue stars
            colors[i3] = 0.7;
            colors[i3 + 1] = 0.8;
            colors[i3 + 2] = 1.0;
        } else if (starType < 0.95) {
            // Yellow stars
            colors[i3] = 1.0;
            colors[i3 + 1] = 0.9;
            colors[i3 + 2] = 0.6;
        } else {
            // Red stars
            colors[i3] = 1.0;
            colors[i3 + 1] = 0.5;
            colors[i3 + 2] = 0.3;
        }
        
        sizes[i] = Math.random() * 2 + 0.5;
    }
    
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));
    
    const material = new THREE.PointsMaterial({
        size: 2,
        vertexColors: true,
        transparent: true,
        opacity: 0.8,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
        sizeAttenuation: true
    });
    
    stars = new THREE.Points(geometry, material);
    scene.add(stars);
}

// Create distant galaxies for depth
function createDistantGalaxies() {
    const galaxyCount = 10;
    
    for (let i = 0; i < galaxyCount; i++) {
        const particleCount = 500;
        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array(particleCount * 3);
        const colors = new Float32Array(particleCount * 3);
        
        // Random galaxy position (far away)
        const centerX = (Math.random() - 0.5) * 1000;
        const centerY = (Math.random() - 0.5) * 500;
        const centerZ = -800 - Math.random() * 400;
        const size = Math.random() * 30 + 20;
        
        for (let j = 0; j < particleCount; j++) {
            const j3 = j * 3;
            
            // Spiral galaxy shape
            const angle = Math.random() * Math.PI * 2;
            const radius = Math.random() * size;
            
            positions[j3] = centerX + Math.cos(angle) * radius;
            positions[j3 + 1] = centerY + (Math.random() - 0.5) * size * 0.2;
            positions[j3 + 2] = centerZ + Math.sin(angle) * radius;
            
            // Distant galaxy color (faint blue/white)
            colors[j3] = 0.6 + Math.random() * 0.4;
            colors[j3 + 1] = 0.7 + Math.random() * 0.3;
            colors[j3 + 2] = 1.0;
        }
        
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        
        const material = new THREE.PointsMaterial({
            size: 1,
            vertexColors: true,
            transparent: true,
            opacity: 0.3,
            blending: THREE.AdditiveBlending,
            depthWrite: false
        });
        
        const distantGalaxy = new THREE.Points(geometry, material);
        scene.add(distantGalaxy);
    }
}

// Animation loop
function animate() {
    requestAnimationFrame(animate);
    time += 0.001;
    
    // Rotate galaxy slowly
    if (galaxyParticles) {
        galaxyParticles.rotation.y += 0.0005;
        galaxyParticles.rotation.x = Math.sin(time * 0.5) * 0.05;
    }
    
    // Rotate nebula in opposite direction
    if (nebulaParticles) {
        nebulaParticles.rotation.y -= 0.0003;
        nebulaParticles.rotation.x = Math.cos(time * 0.3) * 0.03;
    }
    
    // Slow rotation for stars
    if (stars) {
        stars.rotation.y += 0.0001;
    }
    
    // Camera movement based on mouse
    camera.position.x += (mouseX * 0.05 - camera.position.x) * 0.02;
    camera.position.y += (-mouseY * 0.05 - camera.position.y) * 0.02;
    camera.lookAt(scene.position);
    
    // Update shader time
    if (galaxyParticles && galaxyParticles.material.uniforms) {
        galaxyParticles.material.uniforms.time.value = time;
    }
    
    renderer.render(scene, camera);
}

// Mouse movement handler
function onMouseMove(event) {
    mouseX = (event.clientX - window.innerWidth / 2) / 100;
    mouseY = (event.clientY - window.innerHeight / 2) / 100;
}

// Window resize handler
function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}

// Initialize when DOM is loaded
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initGalaxy);
} else {
    initGalaxy();
}