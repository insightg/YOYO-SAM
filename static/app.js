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
// tiles is automatic (24 is default, backend calculates optimal)
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
const exportPanoramasBtn = document.getElementById('export-panoramas-btn');
const classThresholdsGroup = document.getElementById('class-thresholds-group');
const classThresholdsContainer = document.getElementById('class-thresholds');
const resetThresholdsBtn = document.getElementById('reset-thresholds-btn');
const saveThresholdsBtn = document.getElementById('save-thresholds-btn');
const stopDetectionBtn = document.getElementById('stop-detection-btn');

let savedDefaultThresholds = {};  // Loaded from server at startup

// 3D Viewer state
let current3DViewer = null;      // Three.js renderer/scene
let current3DControls = null;    // OrbitControls
let currentPlyUrl = null;
let is3DAvailable = false;       // SAM-3D status
let currentModalDetection = null; // Current detection in modal

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
    loadDefaultThresholds();  // Load saved threshold defaults
    check3DStatus();          // Check SAM-3D availability
    setupEventListeners();
    setupResizableSidebars();
    setup3DEventListeners();  // 3D modal buttons
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

    // Save button (CSV)
    saveImageBtn.addEventListener('click', saveDetectionsCSV);

    // Export CSV button
    exportBtn.addEventListener('click', exportDetectionsCSV);

    // Export Panoramas format button
    exportPanoramasBtn.addEventListener('click', exportPanoramasCSV);

    // Reset per-class thresholds
    resetThresholdsBtn.addEventListener('click', resetClassThresholds);

    // Save thresholds as defaults
    saveThresholdsBtn.addEventListener('click', saveDefaultThresholds);

    // Stop detection button (in progress modal)
    stopDetectionBtn.addEventListener('click', cancelDetection);

    // Modal close handlers
    modalClose.addEventListener('click', closeModal);
    detectionModal.addEventListener('click', (e) => {
        if (e.target === detectionModal) closeModal();
    });

    // Prevent 3D container clicks from closing modal
    const modal3dContainer = document.getElementById('modal-3d-container');
    if (modal3dContainer) {
        modal3dContainer.addEventListener('click', (e) => e.stopPropagation());
    }
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
let imageHasCsv = {};  // Track which images have CSV

async function loadImages() {
    try {
        const response = await fetch('/api/images?limit=5000');
        const data = await response.json();
        allImages = data.images;

        // Build map of images with CSV
        imageHasCsv = {};
        data.images.forEach((img, i) => {
            imageHasCsv[img] = data.has_csv[i];
        });

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

// Update CSV icon for a single image in the list
function updateCsvIcon(imageName) {
    // Update in list view
    const listItem = document.querySelector(`.image-list-item[data-name="${imageName}"]`);
    if (listItem) {
        const csvIcon = listItem.querySelector('.csv-icon');
        if (csvIcon) {
            csvIcon.textContent = imageHasCsv[imageName] ? '📋' : '';
            csvIcon.title = imageHasCsv[imageName] ? 'Has saved detections' : '';
        }
    }

    // Update in thumbnail view
    const thumb = document.querySelector(`.thumbnail[data-name="${imageName}"]`);
    if (thumb) {
        let csvBadge = thumb.querySelector('.csv-badge');
        if (imageHasCsv[imageName]) {
            if (!csvBadge) {
                csvBadge = document.createElement('span');
                csvBadge.className = 'csv-badge';
                csvBadge.textContent = '📋';
                csvBadge.title = 'Has saved detections';
                thumb.insertBefore(csvBadge, thumb.firstChild);
            }
        } else if (csvBadge) {
            csvBadge.remove();
        }
    }
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

            // CSV indicator icon
            const csvIcon = document.createElement('span');
            csvIcon.className = 'csv-icon';
            csvIcon.textContent = imageHasCsv[imageName] ? '📋' : '';
            csvIcon.title = imageHasCsv[imageName] ? 'Has saved detections' : '';

            const nameSpan = document.createElement('span');
            nameSpan.className = 'image-name';
            nameSpan.textContent = imageName;

            item.appendChild(indexSpan);
            item.appendChild(csvIcon);
            item.appendChild(nameSpan);

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

            // CSV indicator for thumbnail
            if (imageHasCsv[imageName]) {
                const csvBadge = document.createElement('span');
                csvBadge.className = 'csv-badge';
                csvBadge.textContent = '📋';
                csvBadge.title = 'Has saved detections';
                thumb.appendChild(csvBadge);
            }

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

    mainImage.onload = async () => {
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

        // Load saved detections if exist
        await loadSavedDetections(imageName);

        // Add 360 panorama button
        addPanoramaButton();
    };
}

// Load saved detections from CSV if exists
async function loadSavedDetections(imageName) {
    try {
        const response = await fetch(`/api/load-detections/${encodeURIComponent(imageName)}`);
        const data = await response.json();

        if (data.exists && data.detections.length > 0) {
            allDetections = data.detections;
            cameraInfo = data.camera;

            // Apply saved default thresholds
            applyDefaultThresholds();

            // Build threshold UI and display
            buildClassThresholdsUI();
            filterAndDisplayDetections();

            // Show detections panel
            detectionsPanel.style.display = 'flex';

            console.log(`Loaded ${allDetections.length} saved detections for ${imageName}`);
        }
    } catch (error) {
        console.error('Error loading saved detections:', error);
    }
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
    exportPanoramasBtn.disabled = true;
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
    exportPanoramasBtn.disabled = detections.length === 0;
}

function buildClassThresholdsUI() {
    // Group subclasses by parent class (original_class)
    const parentClasses = {};  // { parentClass: [subclass1, subclass2, ...] }
    const classCounts = {};    // { class: count }

    allDetections.forEach(d => {
        const parent = d.original_class || d.class;
        const subclass = d.class;

        if (!parentClasses[parent]) {
            parentClasses[parent] = new Set();
        }
        parentClasses[parent].add(subclass);
        classCounts[subclass] = (classCounts[subclass] || 0) + 1;
    });

    const parents = Object.keys(parentClasses).sort();
    if (parents.length === 0) {
        classThresholdsGroup.style.display = 'none';
        return;
    }

    // Assign colors to parent classes
    const parentColors = {};
    parents.forEach((p, i) => {
        parentColors[p] = COLORS[i % COLORS.length];
    });

    const globalThreshold = parseFloat(confidenceSlider.value);
    classThresholdsContainer.innerHTML = '';

    parents.forEach(parent => {
        const subclasses = [...parentClasses[parent]].sort();
        const parentTotal = subclasses.reduce((sum, sc) => sum + (classCounts[sc] || 0), 0);
        const parentThreshold = classThresholds[`__parent__${parent}`] !== undefined
            ? classThresholds[`__parent__${parent}`]
            : globalThreshold;

        // Parent class slider
        const parentItem = document.createElement('div');
        parentItem.className = 'class-threshold-item parent-class';
        parentItem.innerHTML = `
            <div class="class-threshold-color" style="background: ${parentColors[parent]}"></div>
            <span class="class-threshold-name" title="${parent}">${parent}</span>
            <input type="range" class="class-threshold-slider parent-slider" data-parent="${parent}"
                   min="0.1" max="0.9" step="0.05" value="${parentThreshold}">
            <span class="class-threshold-value">${parentThreshold.toFixed(2)}</span>
            <span class="class-threshold-count">${parentTotal}</span>
        `;

        const parentSlider = parentItem.querySelector('.parent-slider');
        const parentValueSpan = parentItem.querySelector('.class-threshold-value');

        parentSlider.addEventListener('input', (e) => {
            const newThreshold = parseFloat(e.target.value);
            classThresholds[`__parent__${parent}`] = newThreshold;
            parentValueSpan.textContent = newThreshold.toFixed(2);

            // Update all subclass sliders
            subclasses.forEach(sc => {
                classThresholds[sc] = newThreshold;
                const scSlider = classThresholdsContainer.querySelector(`.class-threshold-slider[data-class="${sc}"]`);
                if (scSlider) {
                    scSlider.value = newThreshold;
                    scSlider.closest('.class-threshold-item').querySelector('.class-threshold-value').textContent = newThreshold.toFixed(2);
                    const total = classCounts[sc] || 0;
                    const filtered = allDetections.filter(d => d.class === sc && d.score >= newThreshold).length;
                    scSlider.closest('.class-threshold-item').querySelector('.class-threshold-count').textContent = `${filtered}/${total}`;
                }
            });

            filterAndDisplayDetectionsNoRebuild();
        });

        classThresholdsContainer.appendChild(parentItem);

        // Subclass sliders (only if more than one subclass or subclass differs from parent)
        if (subclasses.length > 1 || (subclasses.length === 1 && subclasses[0] !== parent)) {
            subclasses.forEach(sc => {
                const threshold = classThresholds[sc] !== undefined ? classThresholds[sc] : globalThreshold;
                const totalCount = classCounts[sc] || 0;
                const filteredCount = allDetections.filter(d => d.class === sc && d.score >= threshold).length;

                const item = document.createElement('div');
                item.className = 'class-threshold-item subclass';
                item.innerHTML = `
                    <div class="class-threshold-color" style="background: ${parentColors[parent]}; opacity: 0.6"></div>
                    <span class="class-threshold-name subclass-name" title="${sc}">${sc.includes('.') ? sc.split('.').pop() : sc}</span>
                    <input type="range" class="class-threshold-slider" data-class="${sc}"
                           min="0.1" max="0.9" step="0.05" value="${threshold}">
                    <span class="class-threshold-value">${threshold.toFixed(2)}</span>
                    <span class="class-threshold-count">${filteredCount}/${totalCount}</span>
                `;

                const slider = item.querySelector('.class-threshold-slider');
                const valueSpan = item.querySelector('.class-threshold-value');
                const countSpan = item.querySelector('.class-threshold-count');

                slider.addEventListener('input', (e) => {
                    const newThreshold = parseFloat(e.target.value);
                    classThresholds[sc] = newThreshold;
                    valueSpan.textContent = newThreshold.toFixed(2);

                    const newFilteredCount = allDetections.filter(d => d.class === sc && d.score >= newThreshold).length;
                    countSpan.textContent = `${newFilteredCount}/${totalCount}`;

                    filterAndDisplayDetectionsNoRebuild();
                });

                classThresholdsContainer.appendChild(item);
            });
        }
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
    exportPanoramasBtn.disabled = detections.length === 0;
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

// Save current thresholds as defaults
async function saveDefaultThresholds() {
    try {
        const response = await fetch('/api/thresholds', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ thresholds: classThresholds })
        });

        if (!response.ok) throw new Error('Failed to save thresholds');

        savedDefaultThresholds = { ...classThresholds };
        saveThresholdsBtn.textContent = '✓';
        setTimeout(() => { saveThresholdsBtn.textContent = '💾'; }, 1500);

        console.log('Saved default thresholds:', Object.keys(classThresholds).length);
    } catch (error) {
        console.error('Error saving thresholds:', error);
        alert('Error saving thresholds: ' + error.message);
    }
}

// Load saved default thresholds from server
async function loadDefaultThresholds() {
    try {
        const response = await fetch('/api/thresholds');
        const data = await response.json();
        savedDefaultThresholds = data.thresholds || {};
        console.log('Loaded default thresholds:', Object.keys(savedDefaultThresholds).length);
    } catch (error) {
        console.error('Error loading thresholds:', error);
    }
}

// Apply saved default thresholds to current detection
function applyDefaultThresholds() {
    if (Object.keys(savedDefaultThresholds).length === 0) return;

    // Apply saved thresholds for classes that exist in current detection
    const allClasses = new Set(allDetections.map(d => d.class));
    const allParents = new Set(allDetections.map(d => d.original_class || d.class));

    for (const [cls, threshold] of Object.entries(savedDefaultThresholds)) {
        if (allClasses.has(cls) || allParents.has(cls) || cls.startsWith('__parent__')) {
            classThresholds[cls] = threshold;
        }
    }
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

// Current detection EventSource (for cancellation)
let currentEventSource = null;
let isDetecting = false;

async function runDetection() {
    // If already detecting, cancel it
    if (isDetecting && currentEventSource) {
        cancelDetection();
        return;
    }

    if (!currentImage) return;

    const classes = classesTextarea.value
        .split('\n')
        .map(c => c.trim())
        .filter(c => c.length > 0);

    if (classes.length === 0) {
        alert('Please enter at least one class');
        return;
    }

    const tiles = 24;  // Fixed, backend calculates optimal tiling
    // Always request with low confidence (0.1) to get all possible detections
    // Then filter client-side based on slider
    const apiConfidence = 0.1;

    // Show loading with progress
    loadingOverlay.style.display = 'flex';
    resetProgress();
    isDetecting = true;
    detectBtn.disabled = false; // Keep enabled so user can click to stop
    detectBtn.textContent = 'STOP';
    detectBtn.classList.add('stopping');

    // Build SSE URL with query parameters
    const params = new URLSearchParams({
        image_name: currentImage,
        classes: classes.join('|'),
        confidence: apiConfidence,
        tiles: tiles
    });

    currentEventSource = new EventSource(`/api/detect-stream?${params}`);

    currentEventSource.addEventListener('progress', (event) => {
        const data = JSON.parse(event.data);
        updateProgress(data.percent, data.message, data.current_class || '');
    });

    currentEventSource.addEventListener('result', (event) => {
        const data = JSON.parse(event.data);

        // Store all detections (unfiltered) and camera info
        allDetections = data.detections;
        cameraInfo = data.camera;
        classThresholds = {};  // Reset per-class thresholds for new detection

        // Apply saved default thresholds for known classes
        applyDefaultThresholds();

        // Build per-class threshold UI
        buildClassThresholdsUI();

        // Filter and display based on current threshold
        filterAndDisplayDetections();

        // Close connection and hide loading
        finishDetection();
    });

    currentEventSource.addEventListener('error', (event) => {
        let errorMessage = 'Error during detection';
        try {
            const data = JSON.parse(event.data);
            if (data && data.message) {
                errorMessage = data.message;
            }
        } catch (e) {
            // SSE connection error
            if (currentEventSource && currentEventSource.readyState === EventSource.CLOSED) {
                errorMessage = 'Connection closed unexpectedly';
            }
        }

        console.error('Detection error:', errorMessage);
        if (isDetecting) {
            alert('Error during detection: ' + errorMessage);
        }

        finishDetection();
    });

    // Handle SSE connection errors
    currentEventSource.onerror = (event) => {
        // Only handle if not already processed
        if (!currentEventSource || currentEventSource.readyState === EventSource.CLOSED) {
            return;
        }
        console.error('SSE connection error');
        finishDetection();
    };
}

function cancelDetection() {
    console.log('Cancelling detection...');
    if (currentEventSource) {
        currentEventSource.close();
        currentEventSource = null;
    }
    finishDetection();
    updateProgress(0, 'Detection cancelled', '');
}

function finishDetection() {
    if (currentEventSource) {
        currentEventSource.close();
        currentEventSource = null;
    }
    isDetecting = false;
    loadingOverlay.style.display = 'none';
    detectBtn.textContent = 'DETECT';
    detectBtn.classList.remove('stopping');
}

function drawDetections() {
    if (!currentImageInfo) return;

    setupOverlay();

    const ctx = overlay.getContext('2d');
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    const scaleX = overlay.width / currentImageInfo.width;
    const scaleY = overlay.height / currentImageInfo.height;

    // Assign colors by parent class (original_class), sorted for consistency
    const parentColors = {};
    const parentClasses = [...new Set(allDetections.map(d => d.original_class || d.class))].sort();
    parentClasses.forEach((cls, i) => {
        parentColors[cls] = COLORS[i % COLORS.length];
    });

    detections.forEach((det, index) => {
        const [x1, y1, x2, y2] = det.bbox;
        const colorKey = det.original_class || det.class;
        const color = parentColors[colorKey];

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
        const detId = det.id || (index + 1);
        const label = `#${detId} ${labelClass}: ${det.score.toFixed(2)}`;
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

    // Assign colors by parent class (original_class), sorted for consistency
    const parentColors = {};
    const parentClasses = [...new Set(allDetections.map(d => d.original_class || d.class))].sort();
    parentClasses.forEach((cls, i) => {
        parentColors[cls] = COLORS[i % COLORS.length];
    });

    detections.forEach((det, index) => {
        const item = document.createElement('div');
        item.className = 'detection-item';

        // Build GPS info if available (check both formats: det.latitude or det.gps.lat)
        let gpsInfo = '';
        const gpsLat = det.latitude ?? det.gps?.lat;
        const gpsLon = det.longitude ?? det.gps?.lon;
        if (gpsLat && gpsLon) {
            const distInfo = det.distance_m ? `${det.distance_m}m` : '';
            const mapsUrl = `https://www.google.com/maps?q=${gpsLat},${gpsLon}`;
            gpsInfo = `<a href="${mapsUrl}" target="_blank" class="detection-gps" title="${gpsLat.toFixed(6)}, ${gpsLon.toFixed(6)}">📍${distInfo}</a>`;
        }

        // Analysis indicator (deep = LLM, local = GTSRB/RDD model)
        let analysisBadge = '';
        if (det.deep_analyzed) {
            analysisBadge = `<span class="detection-deep" title="Analyzed by LLM (Gemini)">AI</span>`;
        } else if (det.local_analyzed) {
            // Use local_module field if present, otherwise fallback to 'local'
            const moduleName = det.local_module || 'local';
            const localConf = det.local_confidence ? ` (${(det.local_confidence * 100).toFixed(0)}%)` : '';
            analysisBadge = `<span class="detection-local" title="Analyzed by ${moduleName} model${localConf}">${moduleName}</span>`;
        }

        // Display class name (full, CSS handles overflow)
        const displayClass = det.class;
        const colorKey = det.original_class || det.class;

        item.innerHTML = `
            <span class="detection-id">#${det.id || index + 1}</span>
            <div class="detection-color" style="background: ${parentColors[colorKey]}"></div>
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
async function saveDetectionsCSV() {
    if (!currentImage || detections.length === 0) {
        alert('No detections to save');
        return;
    }

    try {
        saveImageBtn.disabled = true;
        saveImageBtn.textContent = 'SAVING...';

        // Save to server
        const response = await fetch('/api/save-detections', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image_name: currentImage,
                detections: detections,
                camera: cameraInfo
            })
        });

        if (!response.ok) {
            throw new Error('Failed to save to server');
        }

        const data = await response.json();
        console.log(`Saved ${data.detections_count} detections to ${data.csv_path}`);

        // Update CSV icon for this image
        imageHasCsv[currentImage] = true;
        updateCsvIcon(currentImage);

        // Also download locally
        const headers = [
            'id', 'image', 'class', 'original_class', 'score',
            'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'bbox_width', 'bbox_height',
            'gps_lat', 'gps_lon', 'camera_lat', 'camera_lon', 'camera_heading',
            'deep_analyzed', 'local_analyzed', 'local_confidence'
        ];

        const rows = detections.map((det, idx) => {
            const [x1, y1, x2, y2] = det.bbox;
            return [
                det.id || (idx + 1),
                currentImage,
                det.class || '',
                det.original_class || '',
                det.score?.toFixed(4) || '',
                Math.round(x1), Math.round(y1), Math.round(x2), Math.round(y2),
                Math.round(x2 - x1), Math.round(y2 - y1),
                det.gps?.lat?.toFixed(6) || '',
                det.gps?.lon?.toFixed(6) || '',
                cameraInfo?.lat?.toFixed(6) || '',
                cameraInfo?.lon?.toFixed(6) || '',
                cameraInfo?.heading?.toFixed(1) || '',
                det.deep_analyzed ? 'true' : 'false',
                det.local_analyzed ? 'true' : 'false',
                det.local_confidence?.toFixed(4) || ''
            ];
        });

        const csvContent = [headers.join(','), ...rows.map(r => r.join(','))].join('\n');
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `${currentImage.replace(/\.[^/.]+$/, '')}_detections.csv`;
        link.click();
        URL.revokeObjectURL(url);

    } catch (error) {
        console.error('Error saving CSV:', error);
        alert('Error saving CSV: ' + error.message);
    } finally {
        saveImageBtn.disabled = false;
        saveImageBtn.textContent = 'SAVE';
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

        // Get color for this class (use parent class, sorted for consistency)
        const parentClasses = [...new Set(allDetections.map(d => d.original_class || d.class))].sort();
        const colorKey = det.original_class || det.class;
        const classIndex = parentClasses.indexOf(colorKey);
        const color = COLORS[classIndex >= 0 ? classIndex % COLORS.length : 0];

        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.strokeRect(relX1, relY1, relX2 - relX1, relY2 - relY1);

        // Update modal info
        const detId = det.id || (index + 1);
        modalTitle.textContent = `#${detId} - ${det.class} (${(det.score * 100).toFixed(1)}%)`;

        // Build info text with GPS if available (check both formats)
        const gpsLat = det.latitude ?? det.gps?.lat;
        const gpsLon = det.longitude ?? det.gps?.lon;
        let infoText = `Size: ${Math.round(cropWidth)} x ${Math.round(cropHeight)} px`;

        if (gpsLat && gpsLon) {
            if (det.distance_m) {
                infoText += ` | Distance: ~${det.distance_m}m`;
            }
            if (det.bearing_deg) {
                infoText += ` | Bearing: ${det.bearing_deg}°`;
            }
            const mapsUrl = `https://www.google.com/maps?q=${gpsLat},${gpsLon}`;
            infoText += `<br><span class="modal-gps">📍 GPS: <a href="${mapsUrl}" target="_blank">${gpsLat.toFixed(6)}, ${gpsLon.toFixed(6)}</a></span>`;
        }

        modalInfo.innerHTML = infoText;

        // Show modal
        detectionModal.style.display = 'flex';

        // Store detection for 3D generation
        currentModalDetection = det;

        // Show 3D button if available
        const generate3dBtn = document.getElementById('generate-3d-btn');
        const toggle3dBtn = document.getElementById('toggle-3d-btn');
        if (is3DAvailable) {
            generate3dBtn.style.display = 'inline-block';
            generate3dBtn.disabled = true;
            generate3dBtn.textContent = 'Verifica...';
            toggle3dBtn.style.display = 'none';

            // Check if 3D already exists in cache
            check3DExists(det).then(result => {
                if (result.exists) {
                    // PLY exists - show view button
                    generate3dBtn.textContent = 'Visualizza 3D';
                    generate3dBtn.disabled = false;
                    generate3dBtn.dataset.cached = 'true';
                    generate3dBtn.dataset.plyUrl = result.ply_url;
                } else {
                    // PLY doesn't exist - show generate button
                    generate3dBtn.textContent = 'Genera 3D';
                    generate3dBtn.disabled = false;
                    generate3dBtn.dataset.cached = 'false';
                    generate3dBtn.dataset.plyUrl = '';
                }
            }).catch(() => {
                generate3dBtn.textContent = 'Genera 3D';
                generate3dBtn.disabled = false;
                generate3dBtn.dataset.cached = 'false';
            });
        } else {
            generate3dBtn.style.display = 'none';
            toggle3dBtn.style.display = 'none';
        }
    };

    // Load the image (use the API endpoint)
    tempImg.src = `/api/image/${encodeURIComponent(currentImage)}?max_size=4000`;
}

/**
 * Open detection modal for any detection (can be called from panorama viewer)
 * @param {Object} det - Detection object with bbox, class, score, etc.
 * @param {string} imageName - Image filename
 * @param {number} imageWidth - Original image width
 * @param {number} imageHeight - Original image height
 * @param {Array} colors - Color palette array
 * @param {Array} allDets - All detections for color consistency
 */
window.openDetectionModalFor = function(det, imageName, imageWidth, imageHeight, colors, allDets) {
    if (!det) return;

    const [x1, y1, x2, y2] = det.bbox;
    const cropWidth = x2 - x1;
    const cropHeight = y2 - y1;

    // Add some padding around the detection (10%)
    const padding = Math.max(cropWidth, cropHeight) * 0.1;
    const padX1 = Math.max(0, x1 - padding);
    const padY1 = Math.max(0, y1 - padding);
    const padX2 = Math.min(imageWidth, x2 + padding);
    const padY2 = Math.min(imageHeight, y2 + padding);

    const finalWidth = padX2 - padX1;
    const finalHeight = padY2 - padY1;

    // Create a temporary image to load the full-res image
    const tempImg = new Image();
    tempImg.crossOrigin = 'anonymous';
    tempImg.onload = () => {
        // Calculate scale for the loaded image (might be resized)
        const scaleX = tempImg.naturalWidth / imageWidth;
        const scaleY = tempImg.naturalHeight / imageHeight;

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
            padX1 * scaleX, padY1 * scaleY,
            finalWidth * scaleX, finalHeight * scaleY,
            0, 0,
            displayWidth, displayHeight
        );

        // Draw bounding box on the crop
        const boxScaleX = displayWidth / finalWidth;
        const boxScaleY = displayHeight / finalHeight;
        const relX1 = (x1 - padX1) * boxScaleX;
        const relY1 = (y1 - padY1) * boxScaleY;
        const relX2 = (x2 - padX1) * boxScaleX;
        const relY2 = (y2 - padY1) * boxScaleY;

        // Get color for this class
        const parentClasses = [...new Set((allDets || [det]).map(d => d.original_class || d.class))].sort();
        const colorKey = det.original_class || det.class;
        const classIndex = parentClasses.indexOf(colorKey);
        const color = (colors || COLORS)[classIndex >= 0 ? classIndex % (colors || COLORS).length : 0];

        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.strokeRect(relX1, relY1, relX2 - relX1, relY2 - relY1);

        // Update modal info
        const detId = det.id || 0;
        modalTitle.textContent = `#${detId} - ${det.class} (${(det.score * 100).toFixed(1)}%)`;

        // Build info text with GPS if available (check both formats)
        const gpsLat = det.latitude ?? det.gps?.lat;
        const gpsLon = det.longitude ?? det.gps?.lon;
        let infoText = `Size: ${Math.round(cropWidth)} x ${Math.round(cropHeight)} px`;

        if (gpsLat && gpsLon) {
            if (det.distance_m) {
                infoText += ` | Distance: ~${det.distance_m}m`;
            }
            if (det.bearing_deg) {
                infoText += ` | Bearing: ${det.bearing_deg}°`;
            }
            const mapsUrl = `https://www.google.com/maps?q=${gpsLat},${gpsLon}`;
            infoText += `<br><span class="modal-gps">📍 GPS: <a href="${mapsUrl}" target="_blank">${gpsLat.toFixed(6)}, ${gpsLon.toFixed(6)}</a></span>`;
        }

        modalInfo.innerHTML = infoText;

        // Show modal
        detectionModal.style.display = 'flex';
    };

    // Load the image
    tempImg.src = `/api/image/${encodeURIComponent(imageName)}?max_size=4000`;
};

let modalCloseBlocked = false;

function closeModal(event) {
    if (modalCloseBlocked) {
        console.log('closeModal BLOCKED - too soon after 3D toggle');
        return;
    }
    console.log('closeModal called!');
    console.log('Event target:', event?.target?.id || event?.target?.className || 'unknown');
    console.log('Event type:', event?.type || 'no event');
    detectionModal.style.display = 'none';

    // Cleanup 3D viewer
    cleanup3DViewer();
    currentModalDetection = null;

    // Reset 3D UI
    const canvas2d = document.getElementById('modal-canvas');
    const container3d = document.getElementById('modal-3d-container');
    const generate3dBtn = document.getElementById('generate-3d-btn');
    const toggle3dBtn = document.getElementById('toggle-3d-btn');

    if (canvas2d) canvas2d.style.display = 'block';
    if (container3d) container3d.style.display = 'none';
    if (generate3dBtn) generate3dBtn.style.display = 'none';
    if (toggle3dBtn) toggle3dBtn.style.display = 'none';
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

// Export detections in Panoramas app format
async function exportPanoramasCSV() {
    if (!currentImage || detections.length === 0) {
        alert('No detections to export');
        return;
    }

    try {
        exportPanoramasBtn.disabled = true;
        exportPanoramasBtn.textContent = 'EXPORTING...';

        const response = await fetch('/api/export-panoramas-batch', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image_name: currentImage,
                detections: detections,
                camera: cameraInfo
            })
        });

        if (!response.ok) {
            throw new Error('Failed to export');
        }

        // Get CSV content from response
        const csvContent = await response.text();

        // Create download
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;

        // Generate filename
        const baseName = currentImage.replace(/\.[^.]+$/, '');
        a.download = `${baseName}_panoramas.csv`;

        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        console.log(`Exported ${detections.length} detections in Panoramas format`);

    } catch (error) {
        console.error('Error exporting Panoramas CSV:', error);
        alert('Error exporting: ' + error.message);
    } finally {
        exportPanoramasBtn.disabled = false;
        exportPanoramasBtn.textContent = 'EXPORT PANORAMAS';
    }
}

// ============ 3D Viewer Functions ============

async function check3DStatus() {
    try {
        const response = await fetch('/api/3d-status');
        const data = await response.json();
        is3DAvailable = data.available;
        console.log('3D Status:', data.message);
    } catch (error) {
        is3DAvailable = false;
        console.log('3D not available:', error);
    }
}

async function check3DExists(det) {
    const response = await fetch('/api/check-3d', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            image_name: currentImage,
            bbox: det.bbox,
            detection_id: String(det.id || '0')
        })
    });
    if (!response.ok) throw new Error('Check failed');
    return await response.json();
}

function setup3DEventListeners() {
    const generate3dBtn = document.getElementById('generate-3d-btn');
    const toggle3dBtn = document.getElementById('toggle-3d-btn');

    if (generate3dBtn) {
        // Left click - view/generate
        generate3dBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            generate3DModel(false);
        });
        // Right click - force regenerate
        generate3dBtn.addEventListener('contextmenu', (e) => {
            e.preventDefault();
            e.stopPropagation();
            if (generate3dBtn.dataset.cached === 'true') {
                if (confirm('Rigenerare il modello 3D? (sovrascriverà quello esistente)')) {
                    generate3DModel(true); // force = true
                }
            }
        });
    }

    if (toggle3dBtn) {
        toggle3dBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            toggle3DView();
        });
    }
}

async function generate3DModel(force = false) {
    const btn = document.getElementById('generate-3d-btn');
    if (!currentModalDetection || !currentImage) return;

    // If cached and not forcing, just load the existing PLY
    const isCached = btn.dataset.cached === 'true';
    const cachedUrl = btn.dataset.plyUrl;

    if (isCached && !force && cachedUrl) {
        btn.disabled = true;
        btn.textContent = 'Caricamento...';

        try {
            console.log('Loading cached 3D from:', cachedUrl);
            currentPlyUrl = cachedUrl;
            await init3DViewer(currentPlyUrl);
            console.log('3D viewer initialized successfully');

            document.getElementById('toggle-3d-btn').style.display = 'inline-block';
            btn.textContent = 'Rigenera 3D';
            btn.disabled = false;
            console.log('About to toggle 3D view');
            modalCloseBlocked = true;
            toggle3DView(true);
            console.log('3D view toggled');
            setTimeout(() => {
                modalCloseBlocked = false;
                console.log('Modal close unblocked');
            }, 2000);
            return;
        } catch (error) {
            console.error('Error loading cached 3D:', error);
            // Fall through to regenerate
        }
    }

    btn.disabled = true;
    btn.textContent = 'Generazione...';

    try {
        const response = await fetch('/api/generate-3d', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image_name: currentImage,
                bbox: currentModalDetection.bbox,
                detection_id: String(currentModalDetection.id || '0'),
                force: force
            })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Generation failed');
        }

        const data = await response.json();
        currentPlyUrl = data.ply_url;

        // Initialize 3D viewer
        await init3DViewer(currentPlyUrl);

        // Show toggle button
        document.getElementById('toggle-3d-btn').style.display = 'inline-block';
        btn.textContent = 'Rigenera 3D';
        btn.dataset.cached = 'true';
        btn.dataset.plyUrl = data.ply_url;

        // Auto-switch to 3D view
        toggle3DView(true);

    } catch (error) {
        console.error('3D generation error:', error);
        btn.textContent = 'Errore';
        alert('Errore generazione 3D: ' + error.message);
        setTimeout(() => {
            // Restore appropriate button text
            btn.textContent = btn.dataset.cached === 'true' ? 'Visualizza 3D' : 'Genera 3D';
            btn.disabled = false;
        }, 2000);
    }
}

// Parse Gaussian Splat PLY file and extract positions + colors
async function parseGaussianSplatPLY(url) {
    const response = await fetch(url);
    const arrayBuffer = await response.arrayBuffer();
    const dataView = new DataView(arrayBuffer);
    const decoder = new TextDecoder();

    // Find end of header
    let headerEnd = 0;
    const bytes = new Uint8Array(arrayBuffer);
    for (let i = 0; i < Math.min(bytes.length, 10000); i++) {
        if (bytes[i] === 0x65 && bytes[i+1] === 0x6e && bytes[i+2] === 0x64 &&
            bytes[i+3] === 0x5f && bytes[i+4] === 0x68 && bytes[i+5] === 0x65 &&
            bytes[i+6] === 0x61 && bytes[i+7] === 0x64 && bytes[i+8] === 0x65 &&
            bytes[i+9] === 0x72) {  // "end_header"
            headerEnd = i + 11;  // Skip "end_header\n"
            break;
        }
    }

    // Parse header
    const headerText = decoder.decode(bytes.slice(0, headerEnd));
    console.log('PLY Header:', headerText);

    // Extract vertex count
    const vertexMatch = headerText.match(/element vertex (\d+)/);
    const vertexCount = vertexMatch ? parseInt(vertexMatch[1]) : 0;
    console.log('Vertex count:', vertexCount);

    // Parse properties and build byte offsets
    const properties = [];
    const propRegex = /property (\w+) (\w+)/g;
    let match;
    let offset = 0;
    const typeSize = { float: 4, uchar: 1, int: 4, double: 8 };

    while ((match = propRegex.exec(headerText)) !== null) {
        const type = match[1];
        const name = match[2];
        properties.push({ name, type, offset });
        offset += typeSize[type] || 4;
    }
    const vertexSize = offset;
    console.log('Properties:', properties.map(p => p.name).join(', '));
    console.log('Vertex size:', vertexSize, 'bytes');

    // Find property indices
    const getPropOffset = (name) => {
        const prop = properties.find(p => p.name === name);
        return prop ? prop.offset : -1;
    };

    const xOffset = getPropOffset('x');
    const yOffset = getPropOffset('y');
    const zOffset = getPropOffset('z');
    const fdc0Offset = getPropOffset('f_dc_0');
    const fdc1Offset = getPropOffset('f_dc_1');
    const fdc2Offset = getPropOffset('f_dc_2');
    const opacityOffset = getPropOffset('opacity');

    console.log('Offsets - x:', xOffset, 'y:', yOffset, 'z:', zOffset,
                'f_dc_0:', fdc0Offset, 'f_dc_1:', fdc1Offset, 'f_dc_2:', fdc2Offset,
                'opacity:', opacityOffset);

    // Read vertex data
    const positions = new Float32Array(vertexCount * 3);
    const colors = new Float32Array(vertexCount * 3);

    // Spherical harmonics C0 coefficient
    const SH_C0 = 0.28209479177387814;

    let dataOffset = headerEnd;
    for (let i = 0; i < vertexCount; i++) {
        const base = dataOffset + i * vertexSize;

        // Position
        positions[i * 3] = dataView.getFloat32(base + xOffset, true);
        positions[i * 3 + 1] = dataView.getFloat32(base + yOffset, true);
        positions[i * 3 + 2] = dataView.getFloat32(base + zOffset, true);

        // Color from spherical harmonics DC component
        // RGB = 0.5 + C0 * f_dc (clamped to [0, 1])
        if (fdc0Offset >= 0) {
            const f_dc_0 = dataView.getFloat32(base + fdc0Offset, true);
            const f_dc_1 = dataView.getFloat32(base + fdc1Offset, true);
            const f_dc_2 = dataView.getFloat32(base + fdc2Offset, true);

            colors[i * 3] = Math.max(0, Math.min(1, 0.5 + SH_C0 * f_dc_0));
            colors[i * 3 + 1] = Math.max(0, Math.min(1, 0.5 + SH_C0 * f_dc_1));
            colors[i * 3 + 2] = Math.max(0, Math.min(1, 0.5 + SH_C0 * f_dc_2));
        } else {
            // Default white if no color data
            colors[i * 3] = 1;
            colors[i * 3 + 1] = 1;
            colors[i * 3 + 2] = 1;
        }
    }

    return { positions, colors, vertexCount };
}

async function init3DViewer(plyUrl) {
    const container = document.getElementById('modal-3d-container');
    const canvas2d = document.getElementById('modal-canvas');

    // Cleanup previous viewer
    cleanup3DViewer();

    // Show 3D container FIRST (so we can get dimensions)
    canvas2d.style.display = 'none';
    container.style.display = 'flex';
    container.style.justifyContent = 'center';
    container.style.alignItems = 'center';

    // Set container size explicitly
    const width = 700;
    const height = 500;
    container.style.width = width + 'px';
    container.style.height = height + 'px';

    // Create scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x00ff00);  // Bright green for debug

    // Add axes helper to verify rendering
    const axesHelper = new THREE.AxesHelper(2);
    scene.add(axesHelper);
    console.log('Scene created with green background and axes helper');

    // Create camera with explicit dimensions
    const camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 1000);
    camera.position.set(0, 0, 3);

    // Create renderer with explicit dimensions
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.domElement.style.border = '3px solid red';  // Debug: visible border
    renderer.domElement.style.display = 'block';
    container.appendChild(renderer.domElement);
    console.log('WebGL canvas appended to container');

    // Add orbit controls
    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 2;

    // Add lights (for potential mesh rendering)
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(5, 5, 5);
    scene.add(directionalLight);

    // Animation loop - start immediately
    let animationId;
    let frameCount = 0;
    function animate() {
        animationId = requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
        frameCount++;
        if (frameCount <= 3) {
            console.log('Animation frame', frameCount, 'rendered');
        }
    }
    animate();
    console.log('Animation loop started');

    // Store references for cleanup
    current3DViewer = { scene, camera, renderer, animationId };
    current3DControls = controls;

    try {
        // Parse Gaussian Splat PLY with custom parser
        console.log('Loading Gaussian Splat PLY:', plyUrl);
        const { positions, colors, vertexCount } = await parseGaussianSplatPLY(plyUrl);

        console.log('Parsed', vertexCount, 'vertices');

        // Debug: check position values
        let posMinX = Infinity, posMaxX = -Infinity;
        let posMinY = Infinity, posMaxY = -Infinity;
        let posMinZ = Infinity, posMaxZ = -Infinity;
        let nanCount = 0;
        for (let i = 0; i < vertexCount; i++) {
            const x = positions[i * 3];
            const y = positions[i * 3 + 1];
            const z = positions[i * 3 + 2];
            if (isNaN(x) || isNaN(y) || isNaN(z)) nanCount++;
            posMinX = Math.min(posMinX, x); posMaxX = Math.max(posMaxX, x);
            posMinY = Math.min(posMinY, y); posMaxY = Math.max(posMaxY, y);
            posMinZ = Math.min(posMinZ, z); posMaxZ = Math.max(posMaxZ, z);
        }
        console.log('Position ranges - X:', posMinX.toFixed(2), 'to', posMaxX.toFixed(2),
                    'Y:', posMinY.toFixed(2), 'to', posMaxY.toFixed(2),
                    'Z:', posMinZ.toFixed(2), 'to', posMaxZ.toFixed(2));
        console.log('NaN positions:', nanCount);

        // Debug: check color values
        let colorSum = 0;
        let minColor = 1, maxColor = 0;
        for (let i = 0; i < Math.min(vertexCount, 1000) * 3; i++) {
            colorSum += colors[i];
            minColor = Math.min(minColor, colors[i]);
            maxColor = Math.max(maxColor, colors[i]);
        }
        console.log('Color stats - avg:', (colorSum / Math.min(vertexCount, 1000) / 3).toFixed(3),
                    'min:', minColor.toFixed(3), 'max:', maxColor.toFixed(3));

        // Create geometry
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

        // Center and scale
        geometry.computeBoundingBox();
        const center = new THREE.Vector3();
        geometry.boundingBox.getCenter(center);
        console.log('Center:', center.x.toFixed(2), center.y.toFixed(2), center.z.toFixed(2));

        geometry.translate(-center.x, -center.y, -center.z);

        // Recompute after centering
        geometry.computeBoundingBox();
        const size = new THREE.Vector3();
        geometry.boundingBox.getSize(size);
        const maxDim = Math.max(size.x, size.y, size.z);
        console.log('Size:', size.x.toFixed(2), size.y.toFixed(2), size.z.toFixed(2), 'Max:', maxDim.toFixed(2));

        if (maxDim > 0) {
            const scale = 2 / maxDim;
            geometry.scale(scale, scale, scale);
        }

        // Calculate point size based on density - use larger size for visibility
        const pointSize = 0.05;  // Fixed large size for debug
        console.log('Point size:', pointSize);

        // Create point cloud - try with fixed white color first for debug
        const material = new THREE.PointsMaterial({
            size: pointSize,
            color: 0xffffff,  // Fixed white for debug
            vertexColors: false,  // Disable vertex colors for debug
            sizeAttenuation: true
        });
        console.log('Material created with fixed white color');

        const points = new THREE.Points(geometry, material);
        scene.add(points);
        console.log('Points added to scene:', vertexCount);

        // Debug: add a wireframe cube (lines) to verify rendering works
        const debugCube = new THREE.Mesh(
            new THREE.BoxGeometry(0.5, 0.5, 0.5),
            new THREE.MeshBasicMaterial({ color: 0xff0000, wireframe: true })
        );
        debugCube.position.set(0, 0, 0);
        scene.add(debugCube);
        console.log('Debug wireframe cube added at origin (size 0.5)');

        // Debug: verify renderer is working
        console.log('Renderer canvas size:', renderer.domElement.width, 'x', renderer.domElement.height);
        console.log('Container display:', getComputedStyle(container).display);

        // Force a render now that everything is added
        renderer.render(scene, camera);
        console.log('Forced render complete');

        // Check WebGL context
        const gl = renderer.getContext();
        console.log('WebGL context valid:', gl !== null);
        console.log('WebGL error:', gl.getError());

        // Check if data is flat (2D-ish) and position camera accordingly
        const isFlat = size.z < size.x * 0.1 && size.z < size.y * 0.1;
        console.log('Is flat:', isFlat, 'Z ratio:', (size.z / Math.max(size.x, size.y)).toFixed(3));

        if (isFlat) {
            // Data is flat on XY plane, view from above
            camera.position.set(0, 0, 3);
        } else {
            // 3D data, view from angle
            camera.position.set(2, 1.5, 2);
        }
        camera.lookAt(0, 0, 0);
        controls.target.set(0, 0, 0);
        controls.update();

    } catch (error) {
        console.error('Error loading Gaussian Splat PLY:', error);
        // Show error in container
        const errorMsg = document.createElement('div');
        errorMsg.style.color = '#ff6666';
        errorMsg.style.padding = '20px';
        errorMsg.textContent = 'Error loading 3D model: ' + error.message;
        container.appendChild(errorMsg);
    }
}

function cleanup3DViewer() {
    const container = document.getElementById('modal-3d-container');

    if (current3DViewer) {
        // Stop animation
        if (current3DViewer.animationId) {
            cancelAnimationFrame(current3DViewer.animationId);
        }

        // Dispose renderer
        if (current3DViewer.renderer) {
            current3DViewer.renderer.dispose();
        }

        // Clear scene
        if (current3DViewer.scene) {
            current3DViewer.scene.traverse((obj) => {
                if (obj.geometry) obj.geometry.dispose();
                if (obj.material) {
                    if (Array.isArray(obj.material)) {
                        obj.material.forEach(m => m.dispose());
                    } else {
                        obj.material.dispose();
                    }
                }
            });
        }

        current3DViewer = null;
    }

    if (current3DControls) {
        current3DControls.dispose();
        current3DControls = null;
    }

    // Clear container
    if (container) {
        container.innerHTML = '';
    }

    currentPlyUrl = null;
}

function toggle3DView(show3D) {
    const canvas2d = document.getElementById('modal-canvas');
    const container3d = document.getElementById('modal-3d-container');
    const toggleBtn = document.getElementById('toggle-3d-btn');

    // If called without argument, toggle
    if (typeof show3D !== 'boolean') {
        show3D = container3d.style.display === 'none';
    }

    if (show3D && currentPlyUrl) {
        canvas2d.style.display = 'none';
        container3d.style.display = 'flex';
        toggleBtn.textContent = '2D';
    } else {
        canvas2d.style.display = 'block';
        container3d.style.display = 'none';
        toggleBtn.textContent = '3D';
    }
}

// ============ Panorama 360 Viewer ============

/**
 * Add 360 panorama button to image container
 */
function addPanoramaButton() {
    const container = document.getElementById('image-container');

    // Remove existing button if any
    const existingBtn = container.querySelector('.open-panorama-btn');
    if (existingBtn) {
        existingBtn.remove();
    }

    // Create button
    const btn = document.createElement('button');
    btn.className = 'open-panorama-btn';
    btn.innerHTML = '360°';
    btn.title = 'Open 360° panoramic view';

    btn.addEventListener('click', (e) => {
        e.stopPropagation();
        openPanoramaViewer();
    });

    container.appendChild(btn);
}

/**
 * Open the panorama viewer with current image and detections
 */
function openPanoramaViewer() {
    if (!currentImage) return;

    // Check if panoramaViewer is initialized
    if (window.panoramaViewer) {
        // Pass current detections and image info to the viewer
        window.panoramaViewer.open(currentImage, {
            detections: detections,
            allDetections: allDetections,
            imageInfo: currentImageInfo,
            colors: COLORS
        });
    } else {
        console.error('PanoramaViewer not initialized');
    }
}

// ============================================================
// TAB SYSTEM & BATCH PROCESSING
// ============================================================

// Batch state
let batchSelectedImages = new Set();
let batchAllImages = [];
let batchEventSource = null;
let batchResults = [];
let batchIsRunning = false;

/**
 * Switch between tabs
 */
function switchTab(tabName) {
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tab === tabName);
    });
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.toggle('active', content.id === `tab-${tabName}`);
    });

    // Load batch image grid when switching to batch tab
    if (tabName === 'batch' && batchAllImages.length === 0) {
        loadBatchImageGrid();
        loadBatchClassLists();
    }
}

/**
 * Initialize tab event listeners
 */
function initTabs() {
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => switchTab(btn.dataset.tab));
    });
}

/**
 * Load class lists into batch dropdown
 */
async function loadBatchClassLists() {
    try {
        const response = await fetch('/api/class-lists');
        const data = await response.json();
        const select = document.getElementById('batch-class-list');

        // Clear existing options except first
        while (select.options.length > 1) {
            select.remove(1);
        }

        // API returns {lists: [...]}
        const lists = data.lists || [];
        lists.forEach(list => {
            const option = document.createElement('option');
            option.value = list.name;
            option.textContent = list.name;
            select.appendChild(option);
        });
    } catch (error) {
        console.error('Error loading class lists:', error);
    }
}

/**
 * Load all images into batch grid
 */
async function loadBatchImageGrid() {
    try {
        const response = await fetch('/api/images?limit=5000');
        const data = await response.json();
        // API returns {images: ["name1.jpg", "name2.jpg", ...]}
        batchAllImages = (data.images || []).map(name => ({ name }));

        const grid = document.getElementById('batch-image-grid');
        grid.innerHTML = batchAllImages.map(img => `
            <div class="batch-thumb" data-image="${img.name}">
                <input type="checkbox" class="batch-checkbox" data-image="${img.name}">
                <img src="/api/thumbnail/${img.name}" loading="lazy" alt="${img.name}">
                <span class="batch-thumb-name">${img.name.length > 20 ? img.name.substring(0, 17) + '...' : img.name}</span>
            </div>
        `).join('');

        // Add click handlers
        grid.querySelectorAll('.batch-thumb').forEach(thumb => {
            thumb.addEventListener('click', (e) => {
                if (e.target.classList.contains('batch-checkbox')) return;
                const checkbox = thumb.querySelector('.batch-checkbox');
                checkbox.checked = !checkbox.checked;
                toggleBatchImage(thumb.dataset.image, checkbox.checked);
            });

            thumb.querySelector('.batch-checkbox').addEventListener('change', (e) => {
                toggleBatchImage(thumb.dataset.image, e.target.checked);
            });
        });

        updateBatchSelectedCount();
    } catch (error) {
        console.error('Error loading batch images:', error);
    }
}

/**
 * Toggle image selection
 */
function toggleBatchImage(imageName, selected) {
    const thumb = document.querySelector(`.batch-thumb[data-image="${imageName}"]`);
    if (selected) {
        batchSelectedImages.add(imageName);
        thumb?.classList.add('selected');
    } else {
        batchSelectedImages.delete(imageName);
        thumb?.classList.remove('selected');
    }
    updateBatchSelectedCount();
}

/**
 * Update selected count display
 */
function updateBatchSelectedCount() {
    const count = batchSelectedImages.size;
    document.getElementById('batch-selected-count').textContent = `${count} selezionate`;
    document.getElementById('batch-start').disabled = count === 0 || batchIsRunning;
}

/**
 * Select all images
 */
function batchSelectAll() {
    batchAllImages.forEach(img => batchSelectedImages.add(img.name));
    document.querySelectorAll('.batch-thumb').forEach(thumb => {
        thumb.classList.add('selected');
        thumb.querySelector('.batch-checkbox').checked = true;
    });
    updateBatchSelectedCount();
}

/**
 * Deselect all images
 */
function batchSelectNone() {
    batchSelectedImages.clear();
    document.querySelectorAll('.batch-thumb').forEach(thumb => {
        thumb.classList.remove('selected');
        thumb.querySelector('.batch-checkbox').checked = false;
    });
    updateBatchSelectedCount();
}

/**
 * Select range of images
 */
function batchSelectRange() {
    const start = prompt('Immagine iniziale (numero):', '1');
    const end = prompt('Immagine finale (numero):', String(batchAllImages.length));

    if (start === null || end === null) return;

    const startIdx = Math.max(0, parseInt(start) - 1);
    const endIdx = Math.min(batchAllImages.length, parseInt(end));

    batchSelectNone();

    for (let i = startIdx; i < endIdx; i++) {
        const img = batchAllImages[i];
        batchSelectedImages.add(img.name);
        const thumb = document.querySelector(`.batch-thumb[data-image="${img.name}"]`);
        if (thumb) {
            thumb.classList.add('selected');
            thumb.querySelector('.batch-checkbox').checked = true;
        }
    }
    updateBatchSelectedCount();
}

/**
 * Start batch processing
 */
async function startBatch() {
    if (batchSelectedImages.size === 0) return;

    const classList = document.getElementById('batch-class-list').value;
    if (!classList) {
        alert('Seleziona una lista di classi');
        return;
    }

    // Get configuration
    const confidence = parseInt(document.getElementById('batch-confidence').value) / 100;
    const tiles = document.getElementById('batch-tiles').value;
    const mode = document.getElementById('batch-mode').value;

    // Load class list content
    let classes = '';
    try {
        const response = await fetch(`/api/class-list/${classList}`);
        const data = await response.json();
        classes = data.content;
    } catch (error) {
        console.error('Error loading class list:', error);
        alert('Errore caricamento lista classi');
        return;
    }

    // Prepare UI
    batchIsRunning = true;
    batchResults = [];
    document.getElementById('batch-start').disabled = true;
    document.getElementById('batch-stop').disabled = false;
    document.getElementById('batch-export-csv').disabled = true;
    document.getElementById('batch-progress').style.display = 'block';
    document.getElementById('batch-log').innerHTML = '';

    const images = Array.from(batchSelectedImages);
    const total = images.length;
    let processed = 0;

    logBatch(`Avvio batch: ${total} immagini, mode: ${mode}`);

    // Build URL with parameters
    const params = new URLSearchParams({
        images: images.join(','),
        classes: classes,
        confidence: confidence,
        tiles: tiles,
        mode: mode
    });

    // Start SSE connection
    const eventSource = new EventSource(`/api/batch-detect-stream?${params}`);
    batchEventSource = eventSource;

    eventSource.onmessage = (event) => {
        // Guard: ignore if we've already stopped
        if (batchEventSource !== eventSource) return;

        const data = JSON.parse(event.data);

        switch (data.type) {
            case 'start':
                logBatch(`[${data.index + 1}/${total}] Elaborazione: ${data.image}`);
                break;

            case 'progress':
                // Update individual image progress if needed
                break;

            case 'complete':
                processed++;
                batchResults.push({
                    image: data.image,
                    detections: data.detections || [],
                    count: data.count
                });
                updateBatchProgress(processed, total);
                logBatch(`✓ ${data.image}: ${data.count} detection${data.count !== 1 ? 's' : ''}`);
                break;

            case 'error':
                processed++;
                updateBatchProgress(processed, total);
                logBatch(`✗ ${data.image}: ${data.error}`, 'error');
                break;

            case 'done':
                eventSource.close();
                if (batchEventSource === eventSource) {
                    batchEventSource = null;
                }
                finishBatch();
                break;
        }
    };

    eventSource.onerror = (error) => {
        // Guard: ignore if we've already stopped or switched to different connection
        if (batchEventSource !== eventSource) return;

        console.error('Batch SSE error:', error);
        eventSource.close();
        batchEventSource = null;
        logBatch('Errore connessione', 'error');

        // Don't call stopBatch to avoid recursion, just update UI
        batchIsRunning = false;
        document.getElementById('batch-start').disabled = batchSelectedImages.size === 0;
        document.getElementById('batch-stop').disabled = true;
        if (batchResults.length > 0) {
            document.getElementById('batch-export-csv').disabled = false;
        }
    };
}

/**
 * Update batch progress bar
 */
function updateBatchProgress(current, total) {
    const percent = Math.round((current / total) * 100);
    document.getElementById('batch-progress-fill').style.width = `${percent}%`;
    document.getElementById('batch-progress-text').textContent = `${current}/${total} immagini`;
    document.getElementById('batch-progress-percent').textContent = `${percent}%`;
}

/**
 * Log message to batch log
 */
function logBatch(message, type = 'info') {
    const log = document.getElementById('batch-log');
    const entry = document.createElement('div');
    entry.className = `log-entry log-${type}`;
    entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
    log.appendChild(entry);
    log.scrollTop = log.scrollHeight;
}

/**
 * Stop batch processing
 */
function stopBatch() {
    if (batchEventSource) {
        batchEventSource.close();
        batchEventSource = null;
    }
    batchIsRunning = false;
    document.getElementById('batch-start').disabled = batchSelectedImages.size === 0;
    document.getElementById('batch-stop').disabled = true;
    logBatch('Batch interrotto');

    if (batchResults.length > 0) {
        document.getElementById('batch-export-csv').disabled = false;
    }
}

/**
 * Finish batch processing
 */
function finishBatch() {
    batchIsRunning = false;
    document.getElementById('batch-start').disabled = batchSelectedImages.size === 0;
    document.getElementById('batch-stop').disabled = true;
    document.getElementById('batch-export-csv').disabled = batchResults.length === 0;

    const totalDetections = batchResults.reduce((sum, r) => sum + r.count, 0);
    logBatch(`Batch completato: ${batchResults.length} immagini, ${totalDetections} detections totali`, 'success');
}

/**
 * Export batch results to CSV
 */
async function exportBatchCsv() {
    if (batchResults.length === 0) return;

    logBatch('Generazione CSV...');

    try {
        const response = await fetch('/api/export-batch-csv', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                results: batchResults
            })
        });

        if (!response.ok) throw new Error('Export failed');

        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `batch_export_${new Date().toISOString().slice(0, 10)}.csv`;
        a.click();
        URL.revokeObjectURL(url);

        logBatch('CSV esportato con successo', 'success');
    } catch (error) {
        console.error('Export error:', error);
        logBatch('Errore esportazione CSV', 'error');
    }
}

/**
 * Initialize batch controls
 */
function initBatchControls() {
    // Selection buttons
    document.getElementById('batch-select-all').addEventListener('click', batchSelectAll);
    document.getElementById('batch-select-none').addEventListener('click', batchSelectNone);
    document.getElementById('batch-select-range').addEventListener('click', batchSelectRange);

    // Action buttons
    document.getElementById('batch-start').addEventListener('click', startBatch);
    document.getElementById('batch-stop').addEventListener('click', stopBatch);
    document.getElementById('batch-export-csv').addEventListener('click', exportBatchCsv);

    // Confidence slider
    const confidenceSlider = document.getElementById('batch-confidence');
    const confidenceValue = document.getElementById('batch-confidence-value');
    confidenceSlider.addEventListener('input', () => {
        confidenceValue.textContent = (parseInt(confidenceSlider.value) / 100).toFixed(2);
    });
}

// Initialize tabs and batch on DOM ready
document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initBatchControls();
});
