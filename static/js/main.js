// Main JavaScript file for the loan prediction system

// Global variables
let currentTheme = localStorage.getItem('theme') || 'light';
let notifications = [];

// DOM Content Loaded
document.addEventListener('DOMContentLoaded', function() {
    // Hide loading screen
    hideLoadingScreen();
    
    // Initialize theme
    initializeTheme();
    
    // Initialize all components
    initializeComponents();
    
    // Initialize tooltips if Bootstrap is loaded
    if (typeof bootstrap !== 'undefined') {
        initializeBootstrapComponents();
    }
    
    // Initialize form handling
    initializeFormHandling();
    
    // Initialize currency formatting
    initializeCurrencyFormatting();
    
    // Initialize navigation
    initializeNavigation();
    
    // Initialize mobile menu
    initializeMobileMenu();
    
    // Initialize charts if present
    initializeCharts();
    
    // Initialize scroll to top button
    initializeScrollToTop();
    
    // Initialize page animations
    initializeAnimations();
});

// Loading Screen Management
function hideLoadingScreen() {
    const loadingScreen = document.getElementById('loadingScreen');
    if (loadingScreen) {
        // Add fade out animation
        loadingScreen.style.opacity = '0';
        loadingScreen.style.transition = 'opacity 0.5s ease-out';
        
        // Remove from DOM after animation
        setTimeout(() => {
            loadingScreen.style.display = 'none';
        }, 500);
    }
}

function showLoadingScreen() {
    const loadingScreen = document.getElementById('loadingScreen');
    if (loadingScreen) {
        loadingScreen.style.display = 'flex';
        loadingScreen.style.opacity = '1';
    }
}

// Fallback to hide loading screen after maximum wait time
setTimeout(hideLoadingScreen, 3000);

// Also hide loading screen when window is fully loaded
window.addEventListener('load', hideLoadingScreen);

// Theme Management
function initializeTheme() {
    const themeToggle = document.getElementById('themeToggle');
    const htmlElement = document.documentElement;
    
    // Set initial theme
    htmlElement.setAttribute('data-theme', currentTheme);
    updateThemeToggleIcon();
    
    // Theme toggle event listener
    if (themeToggle) {
        themeToggle.addEventListener('click', toggleTheme);
    }
}

function toggleTheme() {
    currentTheme = currentTheme === 'light' ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', currentTheme);
    localStorage.setItem('theme', currentTheme);
    updateThemeToggleIcon();
    
    // Show notification
    showNotification(`Switched to ${currentTheme} theme`, 'success');
}

function updateThemeToggleIcon() {
    const themeToggle = document.getElementById('themeToggle');
    if (themeToggle) {
        const icon = themeToggle.querySelector('i');
        if (icon) {
            icon.className = currentTheme === 'light' ? 'fas fa-moon' : 'fas fa-sun';
        }
    }
}

// Component Initialization
function initializeComponents() {
    // Initialize mobile menu toggle
    const mobileToggle = document.querySelector('.navbar-toggler');
    const navbarCollapse = document.querySelector('.navbar-collapse');
    
    if (mobileToggle && navbarCollapse) {
        mobileToggle.addEventListener('click', function() {
            navbarCollapse.classList.toggle('show');
        });
        
        // Close mobile menu when clicking outside
        document.addEventListener('click', function(event) {
            if (!event.target.closest('.navbar')) {
                navbarCollapse.classList.remove('show');
            }
        });
    }
}

function initializeBootstrapComponents() {
    // Initialize tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Initialize popovers
    var popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    var popoverList = popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
}

// Form validation enhancement
function enhanceFormValidation() {
    const forms = document.querySelectorAll('form');
    
    forms.forEach(form => {
        form.addEventListener('submit', function(event) {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
                
                // Focus on first invalid field
                const firstInvalid = form.querySelector(':invalid');
                if (firstInvalid) {
                    firstInvalid.focus();
                    firstInvalid.scrollIntoView({ behavior: 'smooth', block: 'center' });
                }
            }
            
            form.classList.add('was-validated');
        });
        
        // Real-time validation
        const inputs = form.querySelectorAll('input, select, textarea');
        inputs.forEach(input => {
            input.addEventListener('blur', function() {
                if (this.checkValidity()) {
                    this.classList.add('is-valid');
                    this.classList.remove('is-invalid');
                } else {
                    this.classList.add('is-invalid');
                    this.classList.remove('is-valid');
                }
            });
        });
    });
}

// Add loading states to buttons
function addLoadingStates() {
    const buttons = document.querySelectorAll('button[type="submit"], .btn-loading');
    
    buttons.forEach(button => {
        button.addEventListener('click', function() {
            if (this.form && this.form.checkValidity()) {
                showLoadingState(this);
            }
        });
    });
}

function showLoadingState(button) {
    const originalText = button.innerHTML;
    const loadingText = button.dataset.loading || 'Loading...';
    
    button.innerHTML = `<span class="spinner-border spinner-border-sm me-2" role="status"></span>${loadingText}`;
    button.disabled = true;
    
    // Store original text for potential restoration
    button.dataset.originalText = originalText;
}

function hideLoadingState(button) {
    if (button.dataset.originalText) {
        button.innerHTML = button.dataset.originalText;
        button.disabled = false;
    }
}

// Initialize analytics charts
function initializeAnalyticsCharts() {
    // Feature importance chart
    const featureImportanceData = document.getElementById('featureImportanceData');
    if (featureImportanceData) {
        const data = JSON.parse(featureImportanceData.textContent);
        createFeatureImportanceChart(data);
    }
    
    // Model performance chart
    const performanceData = document.getElementById('performanceData');
    if (performanceData) {
        const data = JSON.parse(performanceData.textContent);
        createPerformanceChart(data);
    }
}

function createFeatureImportanceChart(data) {
    const ctx = document.getElementById('featureImportanceChart');
    if (!ctx) return;
    
    new Chart(ctx, {
        type: 'horizontalBar',
        data: {
            labels: data.map(item => item.feature),
            datasets: [{
                label: 'Feature Importance (%)',
                data: data.map(item => item.importance),
                backgroundColor: 'rgba(13, 110, 253, 0.8)',
                borderColor: 'rgba(13, 110, 253, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        }
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return context.parsed.x.toFixed(1) + '%';
                        }
                    }
                }
            }
        }
    });
}

function createPerformanceChart(data) {
    const ctx = document.getElementById('performanceChart');
    if (!ctx) return;
    
    new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            datasets: [{
                label: 'Model Performance',
                data: [
                    data.accuracy * 100,
                    data.precision * 100,
                    data.recall * 100,
                    data.f1_score * 100
                ],
                backgroundColor: 'rgba(13, 110, 253, 0.2)',
                borderColor: 'rgba(13, 110, 253, 1)',
                borderWidth: 2,
                pointBackgroundColor: 'rgba(13, 110, 253, 1)',
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: 'rgba(13, 110, 253, 1)'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        }
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });
}

// Intersection Observer for animations
function observeElements() {
    const options = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate-fade-in-up');
                observer.unobserve(entry.target);
            }
        });
    }, options);
    
    // Observe elements with animation class
    const animateElements = document.querySelectorAll('.animate-on-scroll');
    animateElements.forEach(el => observer.observe(el));
}

// Utility functions
function formatCurrency(amount) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 0,
        maximumFractionDigits: 0
    }).format(amount);
}

function formatPercentage(value) {
    return (value * 100).toFixed(1) + '%';
}

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

// API helper functions
async function makeAPIRequest(url, options = {}) {
    const defaultOptions = {
        headers: {
            'Content-Type': 'application/json',
        }
    };
    
    const config = { ...defaultOptions, ...options };
    
    try {
        const response = await fetch(url, config);
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'API request failed');
        }
        
        return data;
    } catch (error) {
        console.error('API Error:', error);
        throw error;
    }
}

// Show notification
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
    notification.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
    
    notification.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(notification);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (notification.parentNode) {
            notification.remove();
        }
    }, 5000);
}

// Currency Formatting
function initializeCurrencyFormatting() {
    const currencyInputs = document.querySelectorAll('#applicant_income, #coapplicant_income, #loan_amount');
    
    currencyInputs.forEach(input => {
        // Format on input
        input.addEventListener('input', function(e) {
            let value = e.target.value.replace(/,/g, '');
            if (value && !isNaN(value)) {
                // Format with Indian number system (lakhs and crores)
                e.target.value = formatIndianCurrency(parseFloat(value));
            }
        });
        
        // Remove formatting when focused for editing
        input.addEventListener('focus', function(e) {
            let value = e.target.value.replace(/,/g, '');
            e.target.value = value;
        });
        
        // Add formatting back when focus is lost
        input.addEventListener('blur', function(e) {
            let value = e.target.value.replace(/,/g, '');
            if (value && !isNaN(value)) {
                e.target.value = formatIndianCurrency(parseFloat(value));
            }
        });
    });
}

function formatIndianCurrency(amount) {
    // Convert to Indian number format
    if (amount >= 10000000) { // 1 crore
        return (amount / 10000000).toFixed(1) + ' Cr';
    } else if (amount >= 100000) { // 1 lakh
        return (amount / 100000).toFixed(1) + ' L';
    } else if (amount >= 1000) { // thousands
        return (amount / 1000).toFixed(1) + ' K';
    } else {
        return amount.toLocaleString('en-IN');
    }
}

// Mobile Menu Management
function initializeMobileMenu() {
    const mobileMenuToggle = document.getElementById('mobileMenuToggle');
    const navMenu = document.getElementById('navMenu');
    
    if (mobileMenuToggle && navMenu) {
        mobileMenuToggle.addEventListener('click', function() {
            navMenu.classList.toggle('active');
            this.classList.toggle('active');
        });
        
        // Close menu when clicking on a nav link
        const navLinks = navMenu.querySelectorAll('.nav-link');
        navLinks.forEach(link => {
            link.addEventListener('click', function() {
                navMenu.classList.remove('active');
                mobileMenuToggle.classList.remove('active');
            });
        });
        
        // Close menu when clicking outside
        document.addEventListener('click', function(event) {
            if (!navMenu.contains(event.target) && !mobileMenuToggle.contains(event.target)) {
                navMenu.classList.remove('active');
                mobileMenuToggle.classList.remove('active');
            }
        });
    }
}

// Enhanced Button Visibility Fix
function enhanceButtonVisibility() {
    // Force visibility and styling for all buttons
    const buttons = document.querySelectorAll('.btn, button, [type="submit"]');
    buttons.forEach(button => {
        button.style.opacity = '1';
        button.style.visibility = 'visible';
        button.style.pointerEvents = 'auto';
        
        // Add enhanced classes if not present
        if (!button.classList.contains('btn-enhanced')) {
            button.classList.add('btn-enhanced');
        }
    });
    
    // Force visibility for navigation links
    const navLinks = document.querySelectorAll('.nav-link');
    navLinks.forEach(link => {
        link.style.opacity = '1';
        link.style.visibility = 'visible';
        link.style.color = 'var(--text-primary)';
        link.style.fontWeight = '600';
    });
}

// Call enhancement functions
document.addEventListener('DOMContentLoaded', function() {
    setTimeout(enhanceButtonVisibility, 100);
});

// Export functions for use in other scripts
window.LoanSystem = {
    showLoadingState,
    hideLoadingState,
    makeAPIRequest,
    showNotification,
    formatCurrency,
    formatIndianCurrency,
    formatPercentage,
    debounce,
    enhanceButtonVisibility
};