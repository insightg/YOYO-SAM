/**
 * Panoramic 360 Viewer using Pannellum
 * With detection hotspots (labels) and threshold sliders
 */

class PanoramaViewer {
    constructor() {
        this.viewer = null;
        this.isOpen = false;
        this.currentImage = null;
        this.currentIndex = -1;
        this.trajectory = [];
        this.bounds = null;

        // Detection data
        this.allDetections = [];  // Unfiltered
        this.filteredDetections = [];  // After threshold filter
        this.imageInfo = null;
        this.colors = [];
        this.classThresholds = {};  // Per-class thresholds
        this.globalThreshold = 0.3;

        // Minimap
        this.minimapZoom = 1;
        this.minimapCanvas = null;
        this.minimapCtx = null;

        // DOM
        this.container = null;
        this.viewerEl = null;
        this.loadingEl = null;

        // Bind
        this.onKeyDown = this.onKeyDown.bind(this);
    }

    /**
     * Open viewer with image and detections
     */
    async open(imageName, options = {}) {
        console.log('Opening panorama with Pannellum:', imageName);

        this.allDetections = options.allDetections || options.detections || [];
        this.imageInfo = options.imageInfo || null;
        this.colors = options.colors || [];

        // Get DOM elements
        this.viewerEl = document.getElementById('panorama-viewer');
        this.container = document.getElementById('panorama-container');
        this.loadingEl = document.getElementById('panorama-loading');
        this.minimapCanvas = document.getElementById('minimap-canvas');
        this.minimapCtx = this.minimapCanvas.getContext('2d');

        this.isOpen = true;
        this.viewerEl.style.display = 'block';
        this.showLoading(true);

        // Setup event listeners
        this.setupEventListeners();

        // Build threshold UI
        this.buildThresholdUI();

        // Load trajectory for minimap
        if (this.trajectory.length === 0) {
            await this.loadTrajectory();
        }

        // Load panorama
        await this.loadPanorama(imageName);
    }

    /**
     * Setup event listeners
     */
    setupEventListeners() {
        document.getElementById('panorama-close').onclick = () => this.close();
        document.getElementById('nav-prev').onclick = () => this.navigatePrev();
        document.getElementById('nav-next').onclick = () => this.navigateNext();
        document.getElementById('minimap-zoom-in').onclick = () => this.zoomMinimap(1.5);
        document.getElementById('minimap-zoom-out').onclick = () => this.zoomMinimap(0.67);
        this.minimapCanvas.onclick = (e) => this.onMinimapClick(e);
        document.addEventListener('keydown', this.onKeyDown);

        // Controls toggle
        document.getElementById('pano-controls-toggle').onclick = () => {
            const body = document.getElementById('panorama-controls-body');
            const btn = document.getElementById('pano-controls-toggle');
            body.classList.toggle('collapsed');
            btn.textContent = body.classList.contains('collapsed') ? '+' : '−';
        };

        // Global threshold slider
        const globalSlider = document.getElementById('pano-global-threshold');
        const globalValue = document.getElementById('pano-global-value');
        globalSlider.oninput = () => {
            this.globalThreshold = parseFloat(globalSlider.value);
            globalValue.textContent = this.globalThreshold.toFixed(2);
            this.updateHotspots();
        };
    }

    /**
     * Build threshold UI for each class
     */
    buildThresholdUI() {
        const container = document.getElementById('pano-class-thresholds');
        container.innerHTML = '';

        // Get unique classes
        const classes = [...new Set(this.allDetections.map(d => d.original_class || d.class))].sort();

        classes.forEach((cls, index) => {
            const color = this.colors[index % this.colors.length] || '#e94560';
            const threshold = this.classThresholds[cls] || this.globalThreshold;

            const div = document.createElement('div');
            div.className = 'pano-class-slider';
            div.innerHTML = `
                <label>
                    <span class="class-color" style="background: ${color}"></span>
                    <span class="class-name">${cls}</span>
                </label>
                <div class="pano-slider-row">
                    <input type="range" min="0.1" max="0.9" step="0.05" value="${threshold}" data-class="${cls}">
                    <span>${threshold.toFixed(2)}</span>
                </div>
            `;

            const slider = div.querySelector('input');
            const valueSpan = div.querySelector('.pano-slider-row span');

            slider.oninput = () => {
                const val = parseFloat(slider.value);
                this.classThresholds[cls] = val;
                valueSpan.textContent = val.toFixed(2);
                this.updateHotspots();
            };

            container.appendChild(div);
        });
    }

    /**
     * Filter detections based on thresholds
     */
    filterDetections() {
        this.filteredDetections = this.allDetections.filter(det => {
            const cls = det.original_class || det.class;
            const threshold = this.classThresholds[cls] !== undefined
                ? this.classThresholds[cls]
                : this.globalThreshold;
            return det.score >= threshold;
        });

        // Update count
        document.getElementById('pano-detection-count').textContent = `(${this.filteredDetections.length})`;
    }

    /**
     * Close viewer
     */
    close() {
        this.isOpen = false;
        this.viewerEl.style.display = 'none';
        if (this.viewer) {
            this.viewer.destroy();
            this.viewer = null;
        }
        document.removeEventListener('keydown', this.onKeyDown);
    }

    /**
     * Load GPS trajectory
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
     * Load panorama with Pannellum
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

        // Destroy previous viewer
        if (this.viewer) {
            this.viewer.destroy();
            this.viewer = null;
        }

        // Clear container
        this.container.innerHTML = '';

        // Filter detections
        this.filterDetections();

        // Build hotspots from filtered detections
        const hotSpots = this.createHotspots();

        // Create Pannellum viewer
        const panoramaUrl = `/api/panorama/${encodeURIComponent(imageName)}?resolution=high`;

        this.viewer = pannellum.viewer(this.container, {
            type: 'equirectangular',
            panorama: panoramaUrl,
            autoLoad: true,
            showControls: true,
            showFullscreenCtrl: false,
            showZoomCtrl: true,
            mouseZoom: true,
            keyboardZoom: true,
            draggable: true,
            disableKeyboardCtrl: false,
            compass: false,
            northOffset: 0,
            hfov: 100,
            minHfov: 50,
            maxHfov: 120,
            hotSpots: hotSpots,
            hotSpotDebug: false
        });

        // Listen for load event
        this.viewer.on('load', () => {
            console.log('Panorama loaded');
            this.showLoading(false);
            this.drawMinimap();
        });

        this.viewer.on('error', (err) => {
            console.error('Pannellum error:', err);
            this.showLoading(false);
        });
    }

    /**
     * Update hotspots after threshold change
     */
    updateHotspots() {
        if (!this.viewer) return;

        // Remove all existing hotspots
        const currentHotspots = this.viewer.getConfig().hotSpots || [];
        currentHotspots.forEach(hs => {
            try {
                this.viewer.removeHotSpot(hs.id);
            } catch (e) {}
        });

        // Filter and recreate
        this.filterDetections();
        const newHotspots = this.createHotspots();

        newHotspots.forEach(hs => {
            this.viewer.addHotSpot(hs);
        });
    }

    /**
     * Create hotspots from filtered detections (as labels)
     */
    createHotspots() {
        if (!this.filteredDetections || !this.imageInfo) return [];

        const imgWidth = this.imageInfo.width;
        const imgHeight = this.imageInfo.height;

        // Get parent classes for colors
        const parentClasses = [...new Set(this.allDetections.map(d => d.original_class || d.class))].sort();

        const hotSpots = this.filteredDetections.map((det, index) => {
            const [x1, y1, x2, y2] = det.bbox;
            const centerX = (x1 + x2) / 2;
            const centerY = (y1 + y2) / 2;

            // Convert pixel to spherical coordinates
            // Pannellum uses: yaw (-180 to 180), pitch (-90 to 90)
            const yaw = ((centerX / imgWidth) - 0.5) * 360;
            const pitch = (0.5 - (centerY / imgHeight)) * 180;

            // Get color
            const colorKey = det.original_class || det.class;
            const colorIndex = parentClasses.indexOf(colorKey);
            const color = this.colors[colorIndex % this.colors.length] || '#e94560';

            // Short class name
            const shortClass = det.class.includes('.') ?
                det.class.split('.').pop() : det.class;

            const labelText = `#${det.id || index + 1} ${shortClass} ${(det.score * 100).toFixed(0)}%`;

            return {
                id: `det-${det.id || index}`,
                pitch: pitch,
                yaw: yaw,
                type: 'info',
                text: labelText,
                cssClass: 'detection-label-hotspot',
                createTooltipFunc: (hotSpotDiv) => {
                    this.createLabelHotspot(hotSpotDiv, det, color, labelText);
                },
                createTooltipArgs: { detection: det, color: color }
            };
        });

        console.log('Created', hotSpots.length, 'label hotspots');
        return hotSpots;
    }

    /**
     * Create label-style hotspot
     */
    createLabelHotspot(hotSpotDiv, detection, color, labelText) {
        // Make the hotspot a label
        hotSpotDiv.style.cssText = `
            background: ${color};
            color: white;
            padding: 6px 10px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 500;
            white-space: nowrap;
            cursor: pointer;
            box-shadow: 0 2px 8px rgba(0,0,0,0.4);
            border: 2px solid rgba(255,255,255,0.3);
            transition: border-color 0.2s, box-shadow 0.2s;
            pointer-events: auto;
        `;
        hotSpotDiv.textContent = labelText;

        // Hover effect (only border/shadow, no transform)
        hotSpotDiv.onmouseenter = () => {
            hotSpotDiv.style.borderColor = 'white';
            hotSpotDiv.style.boxShadow = '0 4px 16px rgba(0,0,0,0.6)';
        };
        hotSpotDiv.onmouseleave = () => {
            hotSpotDiv.style.borderColor = 'rgba(255,255,255,0.3)';
            hotSpotDiv.style.boxShadow = '0 2px 8px rgba(0,0,0,0.4)';
        };

        // Click to show detection modal
        hotSpotDiv.onclick = (e) => {
            e.stopPropagation();
            e.preventDefault();
            if (window.openDetectionModalFor && this.imageInfo) {
                window.openDetectionModalFor(
                    detection,
                    this.currentImage,
                    this.imageInfo.width,
                    this.imageInfo.height,
                    this.colors,
                    this.allDetections
                );
            }
        };
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
     * Update navigation buttons
     */
    updateNavigationButtons() {
        document.getElementById('nav-prev').disabled = this.currentIndex <= 0;
        document.getElementById('nav-next').disabled = this.currentIndex >= this.trajectory.length - 1;
    }

    /**
     * Navigate to previous
     */
    navigatePrev() {
        if (this.currentIndex > 0) {
            this.loadPanorama(this.trajectory[this.currentIndex - 1].name);
        }
    }

    /**
     * Navigate to next
     */
    navigateNext() {
        if (this.currentIndex < this.trajectory.length - 1) {
            this.loadPanorama(this.trajectory[this.currentIndex + 1].name);
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

        // Draw trajectory line
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

                if (point.heading !== undefined) {
                    const rad = (point.heading - 90) * Math.PI / 180;
                    const arrowLen = 15;
                    ctx.beginPath();
                    ctx.moveTo(pos.x, pos.y);
                    ctx.lineTo(pos.x + Math.cos(rad) * arrowLen, pos.y + Math.sin(rad) * arrowLen);
                    ctx.strokeStyle = 'white';
                    ctx.lineWidth = 2;
                    ctx.stroke();
                }
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
}

// Create global instance
window.panoramaViewer = new PanoramaViewer();
