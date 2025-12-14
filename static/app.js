// SAM3 Object Detection Frontend

// State
let currentImage = null;
let currentImageInfo = null;
let cameraInfo = null;   // Camera GPS position and heading
let allDetections = [];  // All detections from API (unfiltered)
let detections = [];     // Filtered detections (above threshold)
let allImages = [];
let displayedImages = 0;
let classThresholds = {};  // Per-class thresholds: { className: threshold }
let viewMode = 'list';   // 'list' or 'thumbnails'
const BATCH_SIZE = 200;  // Larger batch for list view

// DOM Elements
const thumbnailsContainer = document.getElementById('thumbnails');
const imageListContainer = document.getElementById('image-list');
const imageCountEl = document.getElementById('image-count');
const searchInput = document.getElementById('search-input');
const loadMoreBtn = document.getElementById('load-more-btn');
const viewToggleBtn = document.getElementById('view-toggle');
const sidebarLeft = document.getElementById('sidebar-left');
const sidebarRight = document.getElementById('sidebar-right');
const resizeLeft = document.getElementById('resize-left');
const resizeRight = document.getElementById('resize-right');
const mainImage = document.getElementById('main-image');
const overlay = document.getElementById('overlay');
const placeholder = document.getElementById('placeholder');
const imageInfo = document.getElementById('image-info');
const currentImageName = document.getElementById('current-image-name');
const currentImageSize = document.getElementById('current-image-size');
const detectionsPanel = document.getElementById('detections-panel');
const detectionsList = document.getElementById('detections-list');
const detectionCount = document.getElementById('detection-count');
const confidenceSlider = document.getElementById('confidence');
const confidenceValue = document.getElementById('confidence-value');
const tilesSelect = document.getElementById('tiles');
const classesTextarea = document.getElementById('classes');
const detectBtn = document.getElementById('detect-btn');
const loadingOverlay = document.getElementById('loading-overlay');
const loadingText = document.getElementById('loading-text');
const modelStatus = document.getElementById('model-status');
const classListSelect = document.getElementById('class-list-select');
const loadListBtn = document.getElementById('load-list-btn');
const saveListBtn = document.getElementById('save-list-btn');
const newListBtn = document.getElementById('new-list-btn');
const deleteListBtn = document.getElementById('delete-list-btn');
const saveImageBtn = document.getElementById('save-btn');
const detectionModal = document.getElementById('detection-modal');
const modalClose = document.getElementById('modal-close');
const modalCanvas = document.getElementById('modal-canvas');
const modalTitle = document.getElementById('modal-title');
const modalInfo = document.getElementById('modal-info');
const exportBtn = document.getElementById('export-btn');
const classThresholdsGroup = document.getElementById('class-thresholds-group');
const classThresholdsContainer = document.getElementById('class-thresholds');
const resetThresholdsBtn = document.getElementById('reset-thresholds-btn');

// Color palette for detections
const COLORS = [
    '#e94560', '#4ade80', '#fbbf24', '#3b82f6', '#a855f7',
    '#ec4899', '#14b8a6', '#f97316', '#06b6d4', '#8b5cf6',
    '#ef4444', '#22c55e', '#eab308', '#6366f1', '#d946ef'
];

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    loadImages();
    checkModelStatus();
    loadClassLists();
    setupEventListeners();
    setupResizableSidebars();
});

function setupEventListeners() {
    // Confidence slider - update display and filter detections dynamically
    confidenceSlider.addEventListener('input', (e) => {
        confidenceValue.textContent = e.target.value;
        // If we have detections, filter and redraw them
        if (allDetections.length > 0) {
            filterAndDisplayDetections();
        }
    });

    // Search
    searchInput.addEventListener('input', (e) => {
        filterImages(e.target.value);
    });

    // Load more
    loadMoreBtn.addEventListener('click', () => {
        displayMoreImages();
    });

    // Detect button
    detectBtn.addEventListener('click', runDetection);

    // Class list buttons
    loadListBtn.addEventListener('click', loadSelectedList);
    saveListBtn.addEventListener('click', saveCurrentList);
    newListBtn.addEventListener('click', createNewList);
    deleteListBtn.addEventListener('click', deleteSelectedList);

    // Save image button
    saveImageBtn.addEventListener('click', saveImageWithDetections);

    // Export CSV button
    exportBtn.addEventListener('click', exportDetectionsCSV);

    // Reset per-class thresholds
    resetThresholdsBtn.addEventListener('click', resetClassThresholds);

    // Modal close handlers
    modalClose.addEventListener('click', closeModal);
    detectionModal.addEventListener('click', (e) => {
        if (e.target === detectionModal) closeModal();
    });
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') closeModal();
    });

    // Window resize
    window.addEventListener('resize', () => {
        if (currentImage) {
            drawDetections();
        }
    });

    // View toggle (list/thumbnails)
    viewToggleBtn.addEventListener('click', toggleViewMode);
}

// Toggle between list and thumbnail view
function toggleViewMode() {
    // Remember current selection
    const selectedImage = currentImage;

    if (viewMode === 'list') {
        viewMode = 'thumbnails';
        viewToggleBtn.textContent = '☰';
        viewToggleBtn.title = 'Switch to list view';
        imageListContainer.style.display = 'none';
        thumbnailsContainer.style.display = 'grid';
        // Convert list items to thumbnails
        thumbnailsContainer.innerHTML = '';
        displayedImages = 0;
        displayMoreImages();
    } else {
        viewMode = 'list';
        viewToggleBtn.textContent = '⊞';
        viewToggleBtn.title = 'Switch to thumbnail view';
        thumbnailsContainer.style.display = 'none';
        imageListContainer.style.display = 'block';
        // Convert thumbnails to list items
        imageListContainer.innerHTML = '';
        displayedImages = 0;
        displayMoreImages();
    }

    // Restore selection if there was one
    if (selectedImage) {
        setTimeout(() => {
            const item = viewMode === 'list'
                ? document.querySelector(`.image-list-item[data-name="${selectedImage}"]`)
                : document.querySelector(`.thumbnail[data-name="${selectedImage}"]`);
            if (item) {
                item.classList.add('selected');
                item.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }
        }, 100);
    }
}

// Resizable sidebars
function setupResizableSidebars() {
    let isResizing = false;
    let currentSidebar = null;
    let startX = 0;
    let startWidth = 0;

    function startResize(e, sidebar, isLeft) {
        isResizing = true;
        currentSidebar = sidebar;
        startX = e.clientX;
        startWidth = sidebar.offsetWidth;
        document.body.style.cursor = 'col-resize';
        document.body.style.userSelect = 'none';
        e.target.classList.add('active');
    }

    function doResize(e) {
        if (!isResizing) return;

        const isLeft = currentSidebar === sidebarLeft;
        const diff = e.clientX - startX;
        let newWidth = isLeft ? startWidth + diff : startWidth - diff;

        // Clamp to min/max
        const minWidth = 200;
        const maxWidth = 600;
        newWidth = Math.max(minWidth, Math.min(maxWidth, newWidth));

        currentSidebar.style.width = newWidth + 'px';
    }

    function stopResize() {
        if (!isResizing) return;
        isResizing = false;
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
        document.querySelectorAll('.resize-handle').forEach(h => h.classList.remove('active'));
        currentSidebar = null;

        // Redraw detections if needed
        if (currentImage) {
            setTimeout(drawDetections, 100);
        }
    }

    resizeLeft.addEventListener('mousedown', (e) => startResize(e, sidebarLeft, true));
    resizeRight.addEventListener('mousedown', (e) => startResize(e, sidebarRight, false));
    document.addEventListener('mousemove', doResize);
    document.addEventListener('mouseup', stopResize);
}

// Load images list
async function loadImages() {
    try {
        const response = await fetch('/api/images?limit=5000');
        const data = await response.json();
        allImages = data.images;
        imageCountEl.textContent = `${data.total} images`;
        displayMoreImages();
    } catch (error) {
        console.error('Error loading images:', error);
        imageCountEl.textContent = 'Error loading';
    }
}

function filterImages(query) {
    displayedImages = 0;
    // Clear both containers
    thumbnailsContainer.innerHTML = '';
    imageListContainer.innerHTML = '';

    const filtered = query
        ? allImages.filter(img => img.toLowerCase().includes(query.toLowerCase()))
        : allImages;

    displayImages(filtered.slice(0, BATCH_SIZE), 0);
    displayedImages = Math.min(BATCH_SIZE, filtered.length);

    loadMoreBtn.style.display = displayedImages < filtered.length ? 'block' : 'none';
}

function displayMoreImages() {
    const query = searchInput.value;
    const filtered = query
        ? allImages.filter(img => img.toLowerCase().includes(query.toLowerCase()))
        : allImages;

    const startIdx = displayedImages;
    const nextBatch = filtered.slice(displayedImages, displayedImages + BATCH_SIZE);
    displayImages(nextBatch, startIdx);
    displayedImages += nextBatch.length;

    loadMoreBtn.style.display = displayedImages < filtered.length ? 'block' : 'none';
}

function displayImages(images, startIndex = 0) {
    if (viewMode === 'list') {
        // List view (default) - no thumbnails, just text
        images.forEach((imageName, i) => {
            const item = document.createElement('div');
            item.className = 'image-list-item';
            item.dataset.name = imageName;

            const indexSpan = document.createElement('span');
            indexSpan.className = 'index';
            indexSpan.textContent = (startIndex + i + 1).toString().padStart(4, ' ');

            item.appendChild(indexSpan);
            item.appendChild(document.createTextNode(imageName));

            item.addEventListener('click', () => selectImage(imageName));

            imageListContainer.appendChild(item);
        });
    } else {
        // Thumbnail view - grid with images
        images.forEach(imageName => {
            const thumb = document.createElement('div');
            thumb.className = 'thumbnail';
            thumb.dataset.name = imageName;

            const img = document.createElement('img');
            img.loading = 'lazy';
            img.src = `/api/thumbnail/${encodeURIComponent(imageName)}`;
            img.alt = imageName;

            const nameEl = document.createElement('div');
            nameEl.className = 'name';
            nameEl.textContent = imageName;

            thumb.appendChild(img);
            thumb.appendChild(nameEl);

            thumb.addEventListener('click', () => selectImage(imageName));

            thumbnailsContainer.appendChild(thumb);
        });
    }
}

async function selectImage(imageName) {
    // Update selection UI - handle both list and thumbnail views
    document.querySelectorAll('.thumbnail').forEach(t => t.classList.remove('selected'));
    document.querySelectorAll('.image-list-item').forEach(t => t.classList.remove('selected'));

    const thumb = document.querySelector(`.thumbnail[data-name="${imageName}"]`);
    if (thumb) thumb.classList.add('selected');

    const listItem = document.querySelector(`.image-list-item[data-name="${imageName}"]`);
    if (listItem) listItem.classList.add('selected');

    currentImage = imageName;

    // Load image info
    try {
        const infoResponse = await fetch(`/api/image-info/${encodeURIComponent(imageName)}`);
        currentImageInfo = await infoResponse.json();
    } catch (e) {
        console.error('Error loading image info:', e);
    }

    // Load image
    mainImage.src = `/api/image/${encodeURIComponent(imageName)}?max_size=1600`;
    mainImage.style.display = 'block';
    placeholder.style.display = 'none';

    mainImage.onload = () => {
        // Update image info display
        imageInfo.style.display = 'flex';
        currentImageName.textContent = imageName;
        if (currentImageInfo) {
            currentImageSize.textContent = `${currentImageInfo.width} x ${currentImageInfo.height}`;
        }

        // Clear previous detections
        clearDetections();

        // Enable detect button
        detectBtn.disabled = false;

        // Setup overlay
        setupOverlay();
    };
}

function setupOverlay() {
    const container = document.getElementById('image-container');
    const rect = mainImage.getBoundingClientRect();
    const containerRect = container.getBoundingClientRect();

    // Calculate image position within container
    const imageLeft = rect.left - containerRect.left;
    const imageTop = rect.top - containerRect.top;

    overlay.width = rect.width;
    overlay.height = rect.height;
    overlay.style.left = imageLeft + 'px';
    overlay.style.top = imageTop + 'px';
}

function clearDetections() {
    allDetections = [];
    detections = [];
    cameraInfo = null;
    classThresholds = {};
    const ctx = overlay.getContext('2d');
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    detectionsPanel.style.display = 'none';
    detectionsList.innerHTML = '';
    classThresholdsGroup.style.display = 'none';
    classThresholdsContainer.innerHTML = '';
    saveImageBtn.disabled = true;
    exportBtn.disabled = true;
}

function filterAndDisplayDetections() {
    const globalThreshold = parseFloat(confidenceSlider.value);

    // Filter using per-class thresholds if set, otherwise use global
    detections = allDetections.filter(d => {
        const threshold = classThresholds[d.class] !== undefined
            ? classThresholds[d.class]
            : globalThreshold;
        return d.score >= threshold;
    });

    drawDetections();
    showDetectionsList();
    updateClassThresholdsUI();

    // Enable/disable save and export buttons based on detections
    saveImageBtn.disabled = detections.length === 0;
    exportBtn.disabled = detections.length === 0;
}

function buildClassThresholdsUI() {
    // Get unique classes from all detections
    const classCounts = {};
    allDetections.forEach(d => {
        classCounts[d.class] = (classCounts[d.class] || 0) + 1;
    });

    const classes = Object.keys(classCounts).sort();
    if (classes.length === 0) {
        classThresholdsGroup.style.display = 'none';
        return;
    }

    // Get colors for classes
    const classColors = {};
    classes.forEach((cls, i) => {
        classColors[cls] = COLORS[i % COLORS.length];
    });

    const globalThreshold = parseFloat(confidenceSlider.value);

    classThresholdsContainer.innerHTML = '';

    classes.forEach(cls => {
        const threshold = classThresholds[cls] !== undefined ? classThresholds[cls] : globalThreshold;
        const totalCount = classCounts[cls];
        const filteredCount = allDetections.filter(d => d.class === cls && d.score >= threshold).length;

        const item = document.createElement('div');
        item.className = 'class-threshold-item';
        item.innerHTML = `
            <div class="class-threshold-color" style="background: ${classColors[cls]}"></div>
            <span class="class-threshold-name" title="${cls}">${cls}</span>
            <input type="range" class="class-threshold-slider" data-class="${cls}"
                   min="0.1" max="0.9" step="0.05" value="${threshold}">
            <span class="class-threshold-value">${threshold.toFixed(2)}</span>
            <span class="class-threshold-count">${filteredCount}/${totalCount}</span>
        `;

        const slider = item.querySelector('.class-threshold-slider');
        const valueSpan = item.querySelector('.class-threshold-value');
        const countSpan = item.querySelector('.class-threshold-count');

        slider.addEventListener('input', (e) => {
            const newThreshold = parseFloat(e.target.value);
            classThresholds[cls] = newThreshold;
            valueSpan.textContent = newThreshold.toFixed(2);

            // Update count
            const newFilteredCount = allDetections.filter(d => d.class === cls && d.score >= newThreshold).length;
            countSpan.textContent = `${newFilteredCount}/${totalCount}`;

            // Re-filter and display
            filterAndDisplayDetectionsNoRebuild();
        });

        classThresholdsContainer.appendChild(item);
    });

    classThresholdsGroup.style.display = 'block';
}

function updateClassThresholdsUI() {
    // Update counts in existing UI without rebuilding
    const items = classThresholdsContainer.querySelectorAll('.class-threshold-item');
    const globalThreshold = parseFloat(confidenceSlider.value);

    items.forEach(item => {
        const slider = item.querySelector('.class-threshold-slider');
        const cls = slider.dataset.class;
        const countSpan = item.querySelector('.class-threshold-count');

        const threshold = classThresholds[cls] !== undefined ? classThresholds[cls] : globalThreshold;
        const totalCount = allDetections.filter(d => d.class === cls).length;
        const filteredCount = allDetections.filter(d => d.class === cls && d.score >= threshold).length;

        countSpan.textContent = `${filteredCount}/${totalCount}`;
    });
}

function filterAndDisplayDetectionsNoRebuild() {
    // Filter without rebuilding the class thresholds UI (to avoid loops)
    const globalThreshold = parseFloat(confidenceSlider.value);

    detections = allDetections.filter(d => {
        const threshold = classThresholds[d.class] !== undefined
            ? classThresholds[d.class]
            : globalThreshold;
        return d.score >= threshold;
    });

    drawDetections();
    showDetectionsList();

    saveImageBtn.disabled = detections.length === 0;
    exportBtn.disabled = detections.length === 0;
}

function resetClassThresholds() {
    classThresholds = {};
    const globalThreshold = parseFloat(confidenceSlider.value);

    // Reset all sliders to global value
    const sliders = classThresholdsContainer.querySelectorAll('.class-threshold-slider');
    sliders.forEach(slider => {
        slider.value = globalThreshold;
        const item = slider.closest('.class-threshold-item');
        item.querySelector('.class-threshold-value').textContent = globalThreshold.toFixed(2);
    });

    filterAndDisplayDetections();
}

// DOM Elements for progress
const progressFill = document.getElementById('progress-fill');
const progressPercent = document.getElementById('progress-percent');
const loadingDetail = document.getElementById('loading-detail');

function updateProgress(percent, message, detail = '') {
    if (progressFill) {
        progressFill.style.width = `${percent}%`;
    }
    if (progressPercent) {
        progressPercent.textContent = `${percent}%`;
    }
    if (loadingText && message) {
        loadingText.textContent = message;
    }
    if (loadingDetail) {
        loadingDetail.textContent = detail;
    }
}

function resetProgress() {
    updateProgress(0, 'Inizializzazione...', '');
}

async function runDetection() {
    if (!currentImage) return;

    const classes = classesTextarea.value
        .split('\n')
        .map(c => c.trim())
        .filter(c => c.length > 0);

    if (classes.length === 0) {
        alert('Please enter at least one class');
        return;
    }

    const tiles = parseInt(tilesSelect.value);
    // Always request with low confidence (0.1) to get all possible detections
    // Then filter client-side based on slider
    const apiConfidence = 0.1;

    // Show loading with progress
    loadingOverlay.style.display = 'flex';
    resetProgress();
    detectBtn.disabled = true;

    // Build SSE URL with query parameters
    const params = new URLSearchParams({
        image_name: currentImage,
        classes: classes.join('|'),
        confidence: apiConfidence,
        tiles: tiles
    });

    const eventSource = new EventSource(`/api/detect-stream?${params}`);

    eventSource.addEventListener('progress', (event) => {
        const data = JSON.parse(event.data);
        updateProgress(data.percent, data.message, data.current_class || '');
    });

    eventSource.addEventListener('result', (event) => {
        const data = JSON.parse(event.data);

        // Store all detections (unfiltered) and camera info
        allDetections = data.detections;
        cameraInfo = data.camera;
        classThresholds = {};  // Reset per-class thresholds for new detection

        // Build per-class threshold UI
        buildClassThresholdsUI();

        // Filter and display based on current threshold
        filterAndDisplayDetections();

        // Close connection and hide loading
        eventSource.close();
        loadingOverlay.style.display = 'none';
        detectBtn.disabled = false;
    });

    eventSource.addEventListener('error', (event) => {
        let errorMessage = 'Error during detection';
        try {
            const data = JSON.parse(event.data);
            if (data && data.message) {
                errorMessage = data.message;
            }
        } catch (e) {
            // SSE connection error
            if (eventSource.readyState === EventSource.CLOSED) {
                errorMessage = 'Connection closed unexpectedly';
            }
        }

        console.error('Detection error:', errorMessage);
        alert('Error during detection: ' + errorMessage);

        eventSource.close();
        loadingOverlay.style.display = 'none';
        detectBtn.disabled = false;
    });

    // Handle SSE connection errors
    eventSource.onerror = (event) => {
        // Only handle if not already processed
        if (eventSource.readyState === EventSource.CLOSED) {
            return;
        }
        console.error('SSE connection error');
        eventSource.close();
        loadingOverlay.style.display = 'none';
        detectBtn.disabled = false;
    };
}

function drawDetections() {
    if (!currentImageInfo) return;

    setupOverlay();

    const ctx = overlay.getContext('2d');
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    const scaleX = overlay.width / currentImageInfo.width;
    const scaleY = overlay.height / currentImageInfo.height;

    // Group by class for colors (use original_class for deep analyzed)
    const classColors = {};
    const uniqueClasses = [...new Set(detections.map(d => d.original_class || d.class))];
    uniqueClasses.forEach((cls, i) => {
        classColors[cls] = COLORS[i % COLORS.length];
    });

    detections.forEach((det, index) => {
        const [x1, y1, x2, y2] = det.bbox;
        const colorKey = det.original_class || det.class;
        const color = classColors[colorKey];

        // Scale coordinates
        const sx1 = x1 * scaleX;
        const sy1 = y1 * scaleY;
        const sx2 = x2 * scaleX;
        const sy2 = y2 * scaleY;
        const w = sx2 - sx1;
        const h = sy2 - sy1;

        // Draw box
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(sx1, sy1, w, h);

        // Draw label background (use short label for deep analyzed)
        const labelClass = det.deep_analyzed ? det.class.split('.').pop() : det.class;
        const label = `${labelClass}: ${det.score.toFixed(2)}`;
        ctx.font = '12px sans-serif';
        const textWidth = ctx.measureText(label).width;

        ctx.fillStyle = color;
        ctx.fillRect(sx1, sy1 - 18, textWidth + 8, 18);

        // Draw label text
        ctx.fillStyle = 'white';
        ctx.fillText(label, sx1 + 4, sy1 - 5);
    });
}

function showDetectionsList() {
    detectionsPanel.style.display = 'flex';
    // Show filtered count / total count
    if (allDetections.length > detections.length) {
        detectionCount.textContent = `(${detections.length}/${allDetections.length})`;
    } else {
        detectionCount.textContent = `(${detections.length})`;
    }

    detectionsList.innerHTML = '';

    // Show message if no detections above threshold
    if (detections.length === 0 && allDetections.length > 0) {
        const msg = document.createElement('div');
        msg.className = 'detection-message';
        msg.textContent = `No detections above ${confidenceSlider.value} threshold. Lower the slider to see ${allDetections.length} detection(s).`;
        detectionsList.appendChild(msg);
        return;
    }

    // Group by class for colors (use original_class for deep analyzed)
    const classColors = {};
    const uniqueClasses = [...new Set(detections.map(d => d.original_class || d.class))];
    uniqueClasses.forEach((cls, i) => {
        classColors[cls] = COLORS[i % COLORS.length];
    });

    detections.forEach((det, index) => {
        const item = document.createElement('div');
        item.className = 'detection-item';

        // Build GPS info if available
        let gpsInfo = '';
        if (det.latitude && det.longitude) {
            const distInfo = det.distance_m ? `${det.distance_m}m` : '?';
            gpsInfo = `<span class="detection-gps" title="${det.latitude.toFixed(6)}, ${det.longitude.toFixed(6)}">${distInfo}</span>`;
        }

        // Analysis indicator (deep = LLM, local = GTSRB model)
        let analysisBadge = '';
        if (det.deep_analyzed) {
            analysisBadge = `<span class="detection-deep" title="Analyzed by LLM (Gemini)">AI</span>`;
        } else if (det.local_analyzed) {
            const localConf = det.local_confidence ? ` (${(det.local_confidence * 100).toFixed(0)}%)` : '';
            analysisBadge = `<span class="detection-local" title="Analyzed by GTSRB model${localConf}">GTSRB</span>`;
        }

        // Display class name (shortened if too long)
        const displayClass = det.class.length > 25 ? det.class.substring(0, 22) + '...' : det.class;
        const colorKey = det.original_class || det.class;

        item.innerHTML = `
            <div class="detection-color" style="background: ${classColors[colorKey]}"></div>
            <span class="detection-class" title="${det.class}">${displayClass}</span>
            ${analysisBadge}
            ${gpsInfo}
            <span class="detection-score">${det.score.toFixed(3)}</span>
        `;

        // Highlight on hover
        item.addEventListener('mouseenter', () => highlightDetection(index));
        item.addEventListener('mouseleave', () => drawDetections());

        // Click to open modal with cropped detection
        item.addEventListener('click', () => openDetectionModal(index));

        detectionsList.appendChild(item);
    });
}

function highlightDetection(index) {
    if (!currentImageInfo) return;

    drawDetections();

    const ctx = overlay.getContext('2d');
    const det = detections[index];
    const [x1, y1, x2, y2] = det.bbox;

    const scaleX = overlay.width / currentImageInfo.width;
    const scaleY = overlay.height / currentImageInfo.height;

    const sx1 = x1 * scaleX;
    const sy1 = y1 * scaleY;
    const sx2 = x2 * scaleX;
    const sy2 = y2 * scaleY;
    const w = sx2 - sx1;
    const h = sy2 - sy1;

    // Draw highlighted box
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 4;
    ctx.strokeRect(sx1 - 2, sy1 - 2, w + 4, h + 4);
}

async function checkModelStatus() {
    try {
        const response = await fetch('/api/model-status');
        const data = await response.json();

        const indicator = modelStatus.querySelector('.status-indicator');
        const text = modelStatus.querySelector('.status-text');

        if (data.loaded) {
            indicator.className = 'status-indicator loaded';
            text.textContent = `Model: Ready (${data.device})`;
        } else if (data.loading) {
            indicator.className = 'status-indicator loading';
            text.textContent = 'Model: Loading...';
            // Check again in a few seconds
            setTimeout(checkModelStatus, 3000);
        } else {
            indicator.className = 'status-indicator';
            text.textContent = `Model: Not loaded (${data.device})`;
        }
    } catch (error) {
        console.error('Error checking model status:', error);
    }
}

// Class Lists Management
async function loadClassLists() {
    try {
        const response = await fetch('/api/class-lists');
        const data = await response.json();

        // Clear existing options except the first one
        while (classListSelect.options.length > 1) {
            classListSelect.remove(1);
        }

        // Add options for each list
        data.lists.forEach(list => {
            const option = document.createElement('option');
            option.value = list.name;
            option.textContent = list.name;
            classListSelect.appendChild(option);
        });
    } catch (error) {
        console.error('Error loading class lists:', error);
    }
}

async function loadSelectedList() {
    const listName = classListSelect.value;
    if (!listName) {
        alert('Please select a list first');
        return;
    }

    try {
        const response = await fetch(`/api/class-lists/${encodeURIComponent(listName)}`);
        if (!response.ok) {
            throw new Error('List not found');
        }
        const data = await response.json();
        classesTextarea.value = data.classes.join('\n');
    } catch (error) {
        console.error('Error loading list:', error);
        alert('Error loading list: ' + error.message);
    }
}

async function saveCurrentList() {
    const classes = classesTextarea.value
        .split('\n')
        .map(c => c.trim())
        .filter(c => c.length > 0);

    if (classes.length === 0) {
        alert('No classes to save');
        return;
    }

    // Get name from currently selected or prompt for new name
    let listName = classListSelect.value;
    if (!listName) {
        listName = prompt('Enter a name for the new list:');
        if (!listName) return;
    } else {
        if (!confirm(`Overwrite list "${listName}"?`)) {
            listName = prompt('Enter a name for the new list:');
            if (!listName) return;
        }
    }

    try {
        const response = await fetch('/api/class-lists', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: listName, classes: classes })
        });

        if (!response.ok) {
            throw new Error('Failed to save list');
        }

        const data = await response.json();
        await loadClassLists();
        classListSelect.value = data.name;
        alert(`List "${data.name}" saved with ${data.classes_count} classes`);
    } catch (error) {
        console.error('Error saving list:', error);
        alert('Error saving list: ' + error.message);
    }
}

function createNewList() {
    const listName = prompt('Enter a name for the new list:');
    if (!listName) return;

    // Clear the textarea
    classesTextarea.value = '';
    classListSelect.value = '';

    // Focus on textarea so user can start typing
    classesTextarea.focus();
}

async function deleteSelectedList() {
    const listName = classListSelect.value;
    if (!listName) {
        alert('Please select a list to delete');
        return;
    }

    if (!confirm(`Are you sure you want to delete "${listName}"?`)) {
        return;
    }

    try {
        const response = await fetch(`/api/class-lists/${encodeURIComponent(listName)}`, {
            method: 'DELETE'
        });

        if (!response.ok) {
            throw new Error('Failed to delete list');
        }

        await loadClassLists();
        classListSelect.value = '';
        alert(`List "${listName}" deleted`);
    } catch (error) {
        console.error('Error deleting list:', error);
        alert('Error deleting list: ' + error.message);
    }
}

// Save Image with Detections
async function saveImageWithDetections() {
    if (!currentImage || detections.length === 0) {
        alert('No detections to save');
        return;
    }

    const threshold = parseFloat(confidenceSlider.value);

    try {
        saveImageBtn.disabled = true;
        saveImageBtn.textContent = 'SAVING...';

        const response = await fetch('/api/save-image', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image_name: currentImage,
                detections: detections,
                confidence_threshold: threshold
            })
        });

        if (!response.ok) {
            throw new Error('Failed to save image');
        }

        const data = await response.json();
        alert(`Image saved to:\ntest_out/${data.output_name}\n\n${data.detections_count} detections drawn`);

    } catch (error) {
        console.error('Error saving image:', error);
        alert('Error saving image: ' + error.message);
    } finally {
        saveImageBtn.disabled = false;
        saveImageBtn.textContent = 'SAVE IMAGE';
    }
}

// Detection Modal Functions
function openDetectionModal(index) {
    const det = detections[index];
    if (!det || !currentImageInfo) return;

    const [x1, y1, x2, y2] = det.bbox;
    const cropWidth = x2 - x1;
    const cropHeight = y2 - y1;

    // Add some padding around the detection (10%)
    const padding = Math.max(cropWidth, cropHeight) * 0.1;
    const padX1 = Math.max(0, x1 - padding);
    const padY1 = Math.max(0, y1 - padding);
    const padX2 = Math.min(currentImageInfo.width, x2 + padding);
    const padY2 = Math.min(currentImageInfo.height, y2 + padding);

    const finalWidth = padX2 - padX1;
    const finalHeight = padY2 - padY1;

    // Create a temporary image to load the full-res image
    const tempImg = new Image();
    tempImg.crossOrigin = 'anonymous';
    tempImg.onload = () => {
        // Calculate scale for the loaded image (might be resized)
        const scaleX = tempImg.naturalWidth / currentImageInfo.width;
        const scaleY = tempImg.naturalHeight / currentImageInfo.height;

        // Set canvas size (limit max size for display)
        const maxDisplaySize = 800;
        let displayWidth = finalWidth;
        let displayHeight = finalHeight;

        if (displayWidth > maxDisplaySize || displayHeight > maxDisplaySize) {
            const ratio = Math.min(maxDisplaySize / displayWidth, maxDisplaySize / displayHeight);
            displayWidth *= ratio;
            displayHeight *= ratio;
        }

        modalCanvas.width = displayWidth;
        modalCanvas.height = displayHeight;

        const ctx = modalCanvas.getContext('2d');

        // Draw cropped region
        ctx.drawImage(
            tempImg,
            padX1 * scaleX, padY1 * scaleY,  // Source position
            finalWidth * scaleX, finalHeight * scaleY,  // Source size
            0, 0,  // Destination position
            displayWidth, displayHeight  // Destination size
        );

        // Draw bounding box on the crop
        const boxScaleX = displayWidth / finalWidth;
        const boxScaleY = displayHeight / finalHeight;
        const relX1 = (x1 - padX1) * boxScaleX;
        const relY1 = (y1 - padY1) * boxScaleY;
        const relX2 = (x2 - padX1) * boxScaleX;
        const relY2 = (y2 - padY1) * boxScaleY;

        // Get color for this class
        const uniqueClasses = [...new Set(detections.map(d => d.class))];
        const classIndex = uniqueClasses.indexOf(det.class);
        const color = COLORS[classIndex % COLORS.length];

        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.strokeRect(relX1, relY1, relX2 - relX1, relY2 - relY1);

        // Update modal info
        modalTitle.textContent = `${det.class} (${(det.score * 100).toFixed(1)}%)`;

        // Build info text with GPS if available
        let infoText = `Size: ${Math.round(cropWidth)} x ${Math.round(cropHeight)} px`;

        if (det.latitude && det.longitude) {
            infoText += ` | Distance: ~${det.distance_m}m | Bearing: ${det.bearing_deg}°`;
            infoText += `<br><span class="modal-gps">GPS: <a href="https://www.google.com/maps?q=${det.latitude},${det.longitude}" target="_blank">${det.latitude.toFixed(6)}, ${det.longitude.toFixed(6)}</a></span>`;
        }

        modalInfo.innerHTML = infoText;

        // Show modal
        detectionModal.style.display = 'flex';
    };

    // Load the image (use the API endpoint)
    tempImg.src = `/api/image/${encodeURIComponent(currentImage)}?max_size=4000`;
}

function closeModal() {
    detectionModal.style.display = 'none';
}

// Export detections to CSV with GPS coordinates
function exportDetectionsCSV() {
    if (!currentImage || detections.length === 0) {
        alert('No detections to export');
        return;
    }

    // Build CSV header
    const headers = [
        'image', 'class', 'score',
        'x1', 'y1', 'x2', 'y2',
        'latitude', 'longitude',
        'distance_m', 'bearing_deg', 'geo_confidence'
    ];

    // Add camera info if available
    if (cameraInfo) {
        headers.push('camera_lat', 'camera_lon', 'camera_heading');
    }

    // Build CSV rows
    const rows = detections.map(det => {
        const row = [
            currentImage,
            det.class,
            det.score.toFixed(4),
            det.bbox[0].toFixed(1),
            det.bbox[1].toFixed(1),
            det.bbox[2].toFixed(1),
            det.bbox[3].toFixed(1),
            det.latitude ? det.latitude.toFixed(6) : '',
            det.longitude ? det.longitude.toFixed(6) : '',
            det.distance_m || '',
            det.bearing_deg || '',
            det.geo_confidence || ''
        ];

        if (cameraInfo) {
            row.push(
                cameraInfo.latitude.toFixed(6),
                cameraInfo.longitude.toFixed(6),
                cameraInfo.heading.toFixed(1)
            );
        }

        return row.join(',');
    });

    // Combine header and rows
    const csv = [headers.join(','), ...rows].join('\n');

    // Create download
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;

    // Generate filename
    const baseName = currentImage.replace(/\.[^.]+$/, '');
    const threshold = confidenceSlider.value;
    a.download = `${baseName}_detections_${threshold}.csv`;

    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}
