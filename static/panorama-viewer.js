/**
 * Panoramic 360 Viewer with Detection Overlay
 * Uses Three.js for spherical navigation with detection markers
 */

class PanoramaViewer {
    constructor() {
        // Three.js components
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.sphere = null;
        this.material = null;

        // Detection markers
        this.detectionMarkers = [];
        this.detectionData = null;

        // State
        this.isOpen = false;
        this.currentImage = null;
        this.currentIndex = -1;
        this.trajectory = [];
        this.bounds = null;
        this.imageInfo = null;
        this.colors = [];

        // Minimap state
        this.minimapZoom = 1;

        // DOM elements (cached)
        this.viewer = null;
        this.container = null;
        this.loadingEl = null;
        this.minimapCanvas = null;
        this.minimapCtx = null;

        // Bind methods
        this.animate = this.animate.bind(this);
        this.onResize = this.onResize.bind(this);
        this.onKeyDown = this.onKeyDown.bind(this);
    }

    /**
     * Initialize Three.js scene
     */
    init() {
        console.log('PanoramaViewer: Initializing...');

        // Get DOM elements
        this.viewer = document.getElementById('panorama-viewer');
        this.container = document.getElementById('panorama-container');
        this.loadingEl = document.getElementById('panorama-loading');
        this.minimapCanvas = document.getElementById('minimap-canvas');
        this.minimapCtx = this.minimapCanvas.getContext('2d');

        const width = window.innerWidth;
        const height = window.innerHeight;

        // Scene with dark blue background (so we can see if rendering works)
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x1a1a2e);

        // Camera - positioned at center, looking outward
        this.camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1100);
        this.camera.position.set(0.001, 0, 0.001); // Tiny offset from center

        // Renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(width, height);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.container.appendChild(this.renderer.domElement);

        // OrbitControls - rotate view from center
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableZoom = false;
        this.controls.enablePan = false;
        this.controls.rotateSpeed = -0.25;
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.target.set(0, 0, 0); // Look at center

        // Sphere geometry - inverted for inside view
        const geometry = new THREE.SphereGeometry(500, 64, 32);
        geometry.scale(-1, 1, 1);

        // Material
        this.material = new THREE.MeshBasicMaterial({
            color: 0xffffff,
            side: THREE.DoubleSide
        });

        this.sphere = new THREE.Mesh(geometry, this.material);
        this.scene.add(this.sphere);

        // Event listeners
        this.setupEventListeners();

        console.log('PanoramaViewer: Init complete');
    }

    /**
     * Setup event listeners
     */
    setupEventListeners() {
        document.getElementById('panorama-close').addEventListener('click', () => this.close());
        document.getElementById('nav-prev').addEventListener('click', () => this.navigatePrev());
        document.getElementById('nav-next').addEventListener('click', () => this.navigateNext());
        document.getElementById('minimap-zoom-in').addEventListener('click', () => this.zoomMinimap(1.5));
        document.getElementById('minimap-zoom-out').addEventListener('click', () => this.zoomMinimap(0.67));
        this.minimapCanvas.addEventListener('click', (e) => this.onMinimapClick(e));

        window.addEventListener('resize', this.onResize);
        document.addEventListener('keydown', this.onKeyDown);

        // FOV zoom with mouse wheel
        this.container.addEventListener('wheel', (e) => {
            if (!this.isOpen) return;
            e.preventDefault();
            this.camera.fov = Math.max(30, Math.min(100, this.camera.fov + e.deltaY * 0.05));
            this.camera.updateProjectionMatrix();
        }, { passive: false });
    }

    /**
     * Open viewer with image and optional detections
     */
    async open(imageName, options = {}) {
        console.log('=== Opening panorama ===');
        console.log('Image:', imageName);
        console.log('Options received:', options);
        console.log('Detections count:', options.detections?.length || 0);
        console.log('ImageInfo:', options.imageInfo);
        console.log('Colors:', options.colors?.length || 0);

        if (!this.scene) {
            this.init();
        }

        // Store detection data
        this.detectionData = options.detections || [];
        this.imageInfo = options.imageInfo || null;
        this.colors = options.colors || [];

        this.isOpen = true;
        this.viewer.style.display = 'block';
        this.showLoading(true);

        // Load trajectory for minimap
        if (this.trajectory.length === 0) {
            await this.loadTrajectory();
        }

        // Load panorama texture
        await this.loadPanorama(imageName);

        // Add detection markers
        this.createDetectionMarkers();

        // Start render loop
        this.animate();
    }

    /**
     * Close viewer
     */
    close() {
        this.isOpen = false;
        this.viewer.style.display = 'none';
        this.clearDetectionMarkers();
    }

    /**
     * Load GPS trajectory for minimap
     */
    async loadTrajectory() {
        try {
            const response = await fetch('/api/gps-trajectory');
            const data = await response.json();
            if (data.count > 0) {
                this.trajectory = data.points;
                this.bounds = data.bounds;
            }
        } catch (error) {
            console.error('Failed to load trajectory:', error);
        }
    }

    /**
     * Load panorama texture
     */
    async loadPanorama(imageName) {
        this.currentImage = imageName;
        this.currentIndex = this.trajectory.findIndex(p => p.name === imageName);
        this.updateNavigationButtons();

        // Load info
        try {
            const infoResponse = await fetch(`/api/panorama-info/${encodeURIComponent(imageName)}`);
            const info = await infoResponse.json();
            this.updateInfoPanel(info);
        } catch (e) {
            console.error('Failed to load panorama info:', e);
        }

        // Load texture
        const loader = new THREE.TextureLoader();
        const url = `/api/panorama/${encodeURIComponent(imageName)}?resolution=medium`;

        return new Promise((resolve) => {
            loader.load(url, (texture) => {
                console.log('Texture loaded:', texture.image.width, 'x', texture.image.height);
                texture.colorSpace = THREE.SRGBColorSpace;

                if (this.material.map) {
                    this.material.map.dispose();
                }
                this.material.map = texture;
                this.material.needsUpdate = true;

                this.showLoading(false);
                this.drawMinimap();
                resolve();
            }, undefined, (error) => {
                console.error('Texture load error:', error);
                this.showLoading(false);
                resolve();
            });
        });
    }

    /**
     * Create 3D markers for detections
     */
    createDetectionMarkers() {
        this.clearDetectionMarkers();

        console.log('=== Creating detection markers ===');
        console.log('detectionData:', this.detectionData?.length || 'null');
        console.log('imageInfo:', this.imageInfo);

        if (!this.detectionData || this.detectionData.length === 0) {
            console.log('No detections to display');
            return;
        }

        if (!this.imageInfo) {
            console.log('No imageInfo - cannot calculate positions');
            return;
        }

        const imgWidth = this.imageInfo.width;
        const imgHeight = this.imageInfo.height;

        // Get parent classes for color assignment
        const parentClasses = [...new Set(this.detectionData.map(d => d.original_class || d.class))].sort();

        console.log('Creating markers for', this.detectionData.length, 'detections');
        console.log('Image size:', imgWidth, 'x', imgHeight);

        this.detectionData.forEach((det, index) => {
            const [x1, y1, x2, y2] = det.bbox;
            const centerX = (x1 + x2) / 2;
            const centerY = (y1 + y2) / 2;
            const bboxWidth = x2 - x1;
            const bboxHeight = y2 - y1;

            // Convert pixel coordinates to spherical angles
            // Three.js SphereGeometry UV mapping:
            // - U (horizontal) goes from 0 to 1, mapping to phi from 0 to 2*PI
            // - V (vertical) goes from 0 to 1, mapping to theta from 0 to PI
            // After scale(-1,1,1) inversion, U is mirrored

            // Normalized coordinates (0-1)
            const u = centerX / imgWidth;
            const v = centerY / imgHeight;

            // Convert to spherical angles matching Three.js SphereGeometry
            // phi = azimuth angle (around Y axis), 0 to 2*PI
            // theta = polar angle (from top), 0 to PI
            // Because geometry is inverted with scale(-1,1,1), phi goes backwards
            const phi = (1 - u) * 2 * Math.PI;  // Inverted due to scale(-1,1,1)
            const theta = v * Math.PI;

            // Convert spherical to Cartesian
            // Three.js uses Y-up coordinate system
            const radius = 490; // Slightly inside the texture sphere (radius 500)
            const sinTheta = Math.sin(theta);
            const cosTheta = Math.cos(theta);
            const sinPhi = Math.sin(phi);
            const cosPhi = Math.cos(phi);

            const x = radius * sinTheta * sinPhi;
            const y = radius * cosTheta;
            const z = radius * sinTheta * cosPhi;

            const pos = new THREE.Vector3(x, y, z);

            // Calculate marker size based on bbox angular size
            // Make markers bigger and more visible
            const angularWidth = (bboxWidth / imgWidth) * 2 * Math.PI;
            const angularHeight = (bboxHeight / imgHeight) * Math.PI;
            const markerWidth = Math.max(50, radius * angularWidth);
            const markerHeight = Math.max(50, radius * angularHeight);

            // Get color
            const colorKey = det.original_class || det.class;
            const colorIndex = parentClasses.indexOf(colorKey);
            const colorHex = this.colors[colorIndex % this.colors.length] || '#e94560';

            // Create marker geometry - more visible
            const geometry = new THREE.PlaneGeometry(markerWidth, markerHeight);
            const material = new THREE.MeshBasicMaterial({
                color: colorHex,
                transparent: true,
                opacity: 0.4,
                side: THREE.DoubleSide,
                depthTest: false,
                depthWrite: false
            });

            const marker = new THREE.Mesh(geometry, material);
            marker.position.copy(pos);
            marker.lookAt(0, 0, 0); // Face camera at center

            // Create border with thicker lines
            const borderGeometry = new THREE.PlaneGeometry(markerWidth, markerHeight);
            const edges = new THREE.EdgesGeometry(borderGeometry);
            const lineMaterial = new THREE.LineBasicMaterial({
                color: colorHex,
                linewidth: 3,
                depthTest: false
            });
            const border = new THREE.LineSegments(edges, lineMaterial);
            border.position.copy(pos);
            border.lookAt(0, 0, 0);

            // Create label sprite positioned above the marker
            const label = this.createLabelSprite(det, colorHex, index);
            const labelOffset = new THREE.Vector3().copy(pos).normalize().multiplyScalar(radius - 20);
            labelOffset.y += markerHeight / 2 + 15;
            label.position.copy(pos);
            // Move label slightly toward center and up
            const upDir = new THREE.Vector3(0, 1, 0);
            label.position.addScaledVector(upDir, markerHeight / 2 + 20);

            marker.userData = { detection: det, index: index };

            this.scene.add(marker);
            this.scene.add(border);
            this.scene.add(label);

            this.detectionMarkers.push(marker, border, label);

            if (index < 3) {
                console.log(`Detection ${index}: pixel(${centerX.toFixed(0)}, ${centerY.toFixed(0)}) -> pos(${x.toFixed(1)}, ${y.toFixed(1)}, ${z.toFixed(1)})`);
            }
        });

        console.log('Created', this.detectionMarkers.length / 3, 'detection markers');
    }

    /**
     * Create a text label sprite
     */
    createLabelSprite(detection, color, index) {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = 512;
        canvas.height = 64;

        // Background
        ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Border
        ctx.strokeStyle = color;
        ctx.lineWidth = 4;
        ctx.strokeRect(2, 2, canvas.width - 4, canvas.height - 4);

        // Text
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 28px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';

        const shortClass = detection.class.includes('.') ?
            detection.class.split('.').pop() : detection.class;
        const text = `#${detection.id || index + 1} ${shortClass} (${(detection.score * 100).toFixed(0)}%)`;
        ctx.fillText(text, canvas.width / 2, canvas.height / 2);

        const texture = new THREE.CanvasTexture(canvas);
        const material = new THREE.SpriteMaterial({
            map: texture,
            transparent: true,
            depthTest: false
        });
        const sprite = new THREE.Sprite(material);
        sprite.scale.set(100, 12.5, 1);

        return sprite;
    }

    /**
     * Clear all detection markers
     */
    clearDetectionMarkers() {
        this.detectionMarkers.forEach(marker => {
            this.scene.remove(marker);
            if (marker.geometry) marker.geometry.dispose();
            if (marker.material) {
                if (marker.material.map) marker.material.map.dispose();
                marker.material.dispose();
            }
        });
        this.detectionMarkers = [];
    }

    /**
     * Update info panel
     */
    updateInfoPanel(info) {
        document.getElementById('panorama-name').textContent = info.name || this.currentImage;

        const gpsEl = document.getElementById('panorama-gps');
        if (info.gps && info.gps.lat && info.gps.lon) {
            gpsEl.textContent = `${info.gps.lat.toFixed(6)}, ${info.gps.lon.toFixed(6)}`;
        } else {
            gpsEl.textContent = '';
        }

        const indexEl = document.getElementById('panorama-index');
        if (info.index !== undefined && info.total) {
            indexEl.textContent = `${info.index + 1} / ${info.total}`;
        } else {
            indexEl.textContent = '';
        }
    }

    /**
     * Update navigation button states
     */
    updateNavigationButtons() {
        document.getElementById('nav-prev').disabled = this.currentIndex <= 0;
        document.getElementById('nav-next').disabled = this.currentIndex >= this.trajectory.length - 1;
    }

    /**
     * Navigate to previous panorama
     */
    navigatePrev() {
        if (this.currentIndex > 0) {
            const prevImage = this.trajectory[this.currentIndex - 1].name;
            this.loadPanorama(prevImage).then(() => {
                this.createDetectionMarkers();
            });
        }
    }

    /**
     * Navigate to next panorama
     */
    navigateNext() {
        if (this.currentIndex < this.trajectory.length - 1) {
            const nextImage = this.trajectory[this.currentIndex + 1].name;
            this.loadPanorama(nextImage).then(() => {
                this.createDetectionMarkers();
            });
        }
    }

    /**
     * Draw minimap
     */
    drawMinimap() {
        const canvas = this.minimapCanvas;
        const ctx = this.minimapCtx;
        const width = canvas.width;
        const height = canvas.height;

        ctx.fillStyle = 'rgba(26, 26, 46, 0.95)';
        ctx.fillRect(0, 0, width, height);

        if (!this.bounds || this.trajectory.length === 0) {
            ctx.fillStyle = '#666';
            ctx.font = '12px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('No GPS data', width / 2, height / 2);
            return;
        }

        const padding = 20;
        const latRange = this.bounds.maxLat - this.bounds.minLat;
        const lonRange = this.bounds.maxLon - this.bounds.minLon;
        const scale = Math.min(
            (width - padding * 2) / (lonRange || 0.001),
            (height - padding * 2) / (latRange || 0.001)
        ) * this.minimapZoom;

        const gpsToCanvas = (lat, lon) => ({
            x: padding + (lon - this.bounds.minLon) * scale,
            y: height - padding - (lat - this.bounds.minLat) * scale
        });

        // Draw trajectory
        ctx.beginPath();
        ctx.strokeStyle = 'rgba(233, 69, 96, 0.6)';
        ctx.lineWidth = 2;
        this.trajectory.forEach((point, i) => {
            const pos = gpsToCanvas(point.lat, point.lon);
            if (i === 0) ctx.moveTo(pos.x, pos.y);
            else ctx.lineTo(pos.x, pos.y);
        });
        ctx.stroke();

        // Draw points
        this.trajectory.forEach((point, i) => {
            const pos = gpsToCanvas(point.lat, point.lon);
            if (pos.x < 0 || pos.x > width || pos.y < 0 || pos.y > height) return;

            ctx.beginPath();
            if (i === this.currentIndex) {
                ctx.arc(pos.x, pos.y, 8, 0, Math.PI * 2);
                ctx.fillStyle = '#e94560';
                ctx.fill();
                ctx.strokeStyle = 'white';
                ctx.lineWidth = 2;
                ctx.stroke();
            } else {
                ctx.arc(pos.x, pos.y, 3, 0, Math.PI * 2);
                ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
                ctx.fill();
            }
        });
    }

    /**
     * Handle minimap click
     */
    onMinimapClick(e) {
        if (!this.bounds || this.trajectory.length === 0) return;

        const rect = this.minimapCanvas.getBoundingClientRect();
        const clickX = (e.clientX - rect.left) * (this.minimapCanvas.width / rect.width);
        const clickY = (e.clientY - rect.top) * (this.minimapCanvas.height / rect.height);

        const padding = 20;
        const latRange = this.bounds.maxLat - this.bounds.minLat;
        const lonRange = this.bounds.maxLon - this.bounds.minLon;
        const scale = Math.min(
            (this.minimapCanvas.width - padding * 2) / (lonRange || 0.001),
            (this.minimapCanvas.height - padding * 2) / (latRange || 0.001)
        ) * this.minimapZoom;

        let minDist = Infinity, closestIdx = -1;
        this.trajectory.forEach((point, i) => {
            const px = padding + (point.lon - this.bounds.minLon) * scale;
            const py = this.minimapCanvas.height - padding - (point.lat - this.bounds.minLat) * scale;
            const dist = Math.sqrt((clickX - px) ** 2 + (clickY - py) ** 2);
            if (dist < minDist) {
                minDist = dist;
                closestIdx = i;
            }
        });

        if (closestIdx >= 0 && minDist < 30) {
            this.loadPanorama(this.trajectory[closestIdx].name);
        }
    }

    /**
     * Zoom minimap
     */
    zoomMinimap(factor) {
        this.minimapZoom = Math.max(0.5, Math.min(5, this.minimapZoom * factor));
        this.drawMinimap();
    }

    /**
     * Show/hide loading
     */
    showLoading(show) {
        this.loadingEl.style.display = show ? 'flex' : 'none';
    }

    /**
     * Keyboard handler
     */
    onKeyDown(e) {
        if (!this.isOpen) return;
        switch (e.key) {
            case 'Escape': this.close(); break;
            case 'ArrowLeft': this.navigatePrev(); break;
            case 'ArrowRight': this.navigateNext(); break;
            case '+': case '=': this.zoomMinimap(1.5); break;
            case '-': this.zoomMinimap(0.67); break;
        }
    }

    /**
     * Window resize handler
     */
    onResize() {
        if (!this.isOpen || !this.camera || !this.renderer) return;
        const width = window.innerWidth;
        const height = window.innerHeight;
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }

    /**
     * Animation loop
     */
    animate() {
        if (!this.isOpen) return;
        requestAnimationFrame(this.animate);
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }
}

// Create global instance
window.panoramaViewer = new PanoramaViewer();
