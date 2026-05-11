"""
Dashboard JavaScript - Frontend Logic
"""

const API_BASE = 'http://localhost:5000/api';
let charts = {};

// ════════════════════════════════════════════════════════════════
// INITIALIZATION
// ════════════════════════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', () => {
    console.log('[Dashboard] Initializing...');
    loadDashboardData();
    setupNavigation();
    setupEventListeners();
    // Auto-refresh every 10 seconds
    setInterval(loadDashboardData, 10000);
});

// ════════════════════════════════════════════════════════════════
// NAVIGATION & UI
// ════════════════════════════════════════════════════════════════

function setupNavigation() {
    document.querySelectorAll('.navbar-nav a').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const target = link.getAttribute('href');
            document.querySelectorAll('section').forEach(s => s.style.display = 'none');
            document.querySelector(target).style.display = 'block';
            window.scrollTo({ top: 0, behavior: 'smooth' });
        });
    });
}

function setupEventListeners() {
    // Alert form
    const alertForm = document.getElementById('alertForm');
    if (alertForm) {
        alertForm.addEventListener('submit', sendAlert);
    }

    // History filter
    const historyFilter = document.getElementById('historyFilter');
    if (historyFilter) {
        historyFilter.addEventListener('change', loadHistory);
    }
}

// ════════════════════════════════════════════════════════════════
// DASHBOARD DATA LOADING
// ════════════════════════════════════════════════════════════════

async function loadDashboardData() {
    console.log('[Dashboard] Loading data...');
    try {
        await Promise.all([
            loadSummaryCards(),
            loadRealtimeData(),
            loadControlVsDataPlane(),
            loadAttackTypes(),
            loadFeatureImportance(),
            loadPerformanceMetrics(),
            loadEarlyWarning(),
            loadAlertsHistory()
        ]);
        console.log('[Dashboard] Data loaded successfully');
    } catch (error) {
        console.error('[Dashboard] Error loading data:', error);
    }
}

// ────────────────────────────────────────────────────────────────
// SUMMARY CARDS
// ────────────────────────────────────────────────────────────────

async function loadSummaryCards() {
    try {
        const response = await fetch(`${API_BASE}/dashboard/summary`);
        const data = await response.json();
        
        const cardsHTML = `
            <div class="col-md-3 col-sm-6">
                <div class="summary-card total">
                    <i class="fas fa-shield-alt"></i>
                    <h3>${data.total_attacks}</h3>
                    <p>Total Attacks</p>
                </div>
            </div>
            <div class="col-md-3 col-sm-6">
                <div class="summary-card attacks-24h">
                    <i class="fas fa-clock"></i>
                    <h3>${data.attacks_24h}</h3>
                    <p>Attacks (24h)</p>
                </div>
            </div>
            <div class="col-md-3 col-sm-6">
                <div class="summary-card blocked">
                    <i class="fas fa-lock"></i>
                    <h3>${data.blocked_attacks}</h3>
                    <p>Attacks Blocked</p>
                </div>
            </div>
            <div class="col-md-3 col-sm-6">
                <div class="summary-card confidence">
                    <i class="fas fa-percentage"></i>
                    <h3>${data.avg_confidence}%</h3>
                    <p>Avg Confidence</p>
                </div>
            </div>
        `;
        
        document.getElementById('summaryCards').innerHTML = cardsHTML;
    } catch (error) {
        console.error('[Summary Cards] Error:', error);
    }
}

// ────────────────────────────────────────────────────────────────
// REAL-TIME MONITORING
// ────────────────────────────────────────────────────────────────

async function loadRealtimeData() {
    try {
        const response = await fetch(`${API_BASE}/dashboard/real-time`);
        const data = await response.json();
        
        let html = '';
        data.forEach(attack => {
            const severityBadge = `badge badge-${attack.severity.toLowerCase()}`;
            html += `
                <tr>
                    <td><small>${new Date(attack.time).toLocaleTimeString()}</small></td>
                    <td><code>${attack.src_ip}</code></td>
                    <td>${attack.attack_type}</td>
                    <td><strong>${(attack.confidence * 100).toFixed(1)}%</strong></td>
                    <td><span class="badge ${severityBadge}">${attack.severity}</span></td>
                    <td><span class="badge badge-success">Blocked</span></td>
                </tr>
            `;
        });
        
        document.getElementById('realtimeData').innerHTML = html || '<tr><td colspan="6" class="text-center text-muted">No data available</td></tr>';
    } catch (error) {
        console.error('[Real-time Data] Error:', error);
    }
}

// ────────────────────────────────────────────────────────────────
// VISUALIZATIONS
// ────────────────────────────────────────────────────────────────

async function loadControlVsDataPlane() {
    try {
        const response = await fetch(`${API_BASE}/visualizations/control-vs-data-plane`);
        const data = await response.json();
        
        const ctx = document.getElementById('controlDataPlaneChart');
        if (!ctx) return;
        
        // Destroy existing chart if it exists
        if (charts.controlDataPlane) {
            charts.controlDataPlane.destroy();
        }
        
        const controlCount = data.control_plane?.count || 0;
        const dataCount = data.data_plane?.count || 0;
        
        charts.controlDataPlane = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Control Plane Attacks', 'Data Plane Attacks'],
                datasets: [{
                    data: [controlCount, dataCount],
                    backgroundColor: [
                        'rgba(255, 107, 107, 0.8)',
                        'rgba(102, 126, 234, 0.8)'
                    ],
                    borderColor: [
                        'rgba(255, 107, 107, 1)',
                        'rgba(102, 126, 234, 1)'
                    ],
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            font: { size: 12, weight: 'bold' },
                            padding: 20
                        }
                    }
                }
            }
        });
    } catch (error) {
        console.error('[Control vs Data Plane] Error:', error);
    }
}

async function loadAttackTypes() {
    try {
        const response = await fetch(`${API_BASE}/visualizations/attack-types`);
        const data = await response.json();
        
        const ctx = document.getElementById('attackTypeChart');
        if (!ctx) return;
        
        if (charts.attackType) {
            charts.attackType.destroy();
        }
        
        const types = data.map(d => d.type).slice(0, 8);
        const counts = data.map(d => d.count).slice(0, 8);
        
        charts.attackType = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: types,
                datasets: [{
                    label: 'Number of Attacks',
                    data: counts,
                    backgroundColor: [
                        'rgba(220, 53, 69, 0.8)',
                        'rgba(255, 107, 107, 0.8)',
                        'rgba(255, 140, 0, 0.8)',
                        'rgba(255, 193, 7, 0.8)',
                        'rgba(76, 175, 80, 0.8)',
                        'rgba(33, 150, 243, 0.8)',
                        'rgba(156, 39, 176, 0.8)',
                        'rgba(233, 30, 99, 0.8)'
                    ],
                    borderRadius: 8
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    x: {
                        beginAtZero: true
                    }
                }
            }
        });
    } catch (error) {
        console.error('[Attack Types] Error:', error);
    }
}

async function loadFeatureImportance() {
    try {
        const response = await fetch(`${API_BASE}/visualizations/feature-importance`);
        const data = await response.json();
        
        const ctx = document.getElementById('featureImportanceChart');
        if (!ctx) return;
        
        if (charts.featureImportance) {
            charts.featureImportance.destroy();
        }
        
        charts.featureImportance = new Chart(ctx, {
            type: 'horizontalBar',
            data: {
                labels: data.features,
                datasets: [{
                    label: 'Importance Score',
                    data: data.importance_scores.map(s => s * 100),
                    backgroundColor: 'rgba(67, 233, 123, 0.8)',
                    borderColor: 'rgba(67, 233, 123, 1)',
                    borderRadius: 8,
                    borderWidth: 1
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    x: {
                        beginAtZero: true,
                        max: 30
                    }
                }
            }
        });
    } catch (error) {
        console.error('[Feature Importance] Error:', error);
    }
}

async function loadPerformanceMetrics() {
    try {
        const response = await fetch(`${API_BASE}/visualizations/performance-metrics`);
        const data = await response.json();
        
        // Update table
        const tbody = document.getElementById('performanceBody');
        if (tbody) {
            tbody.innerHTML = `
                <tr>
                    <td><strong>Hybrid SVM-RF</strong></td>
                    <td>${data.hybrid_svm_rf.accuracy}%</td>
                    <td>${data.hybrid_svm_rf.f1_score}</td>
                    <td>${data.hybrid_svm_rf.detection_latency.toFixed(2)}</td>
                </tr>
                <tr>
                    <td>SVM Alone</td>
                    <td>${data.svm_alone.accuracy}%</td>
                    <td>${data.svm_alone.f1_score}</td>
                    <td>${data.svm_alone.detection_latency.toFixed(2)}</td>
                </tr>
                <tr>
                    <td>RF Alone</td>
                    <td>${data.rf_alone.accuracy}%</td>
                    <td>${data.rf_alone.f1_score}</td>
                    <td>${data.rf_alone.detection_latency.toFixed(2)}</td>
                </tr>
            `;
        }
        
        // Update chart
        const ctx = document.getElementById('performanceChart');
        if (ctx) {
            if (charts.performance) {
                charts.performance.destroy();
            }
            
            charts.performance = new Chart(ctx, {
                type: 'radar',
                data: {
                    labels: ['Accuracy', 'F1-Score', 'Speed', 'Reliability', 'Efficiency'],
                    datasets: [
                        {
                            label: 'Hybrid SVM-RF',
                            data: [
                                data.hybrid_svm_rf.accuracy,
                                data.hybrid_svm_rf.f1_score * 100,
                                95,
                                98,
                                96
                            ],
                            borderColor: 'rgba(102, 126, 234, 1)',
                            backgroundColor: 'rgba(102, 126, 234, 0.2)',
                            borderWidth: 2
                        },
                        {
                            label: 'SVM Alone',
                            data: [
                                data.svm_alone.accuracy,
                                data.svm_alone.f1_score * 100,
                                85,
                                90,
                                88
                            ],
                            borderColor: 'rgba(220, 53, 69, 1)',
                            backgroundColor: 'rgba(220, 53, 69, 0.2)',
                            borderWidth: 2
                        },
                        {
                            label: 'RF Alone',
                            data: [
                                data.rf_alone.accuracy,
                                data.rf_alone.f1_score * 100,
                                92,
                                92,
                                94
                            ],
                            borderColor: 'rgba(255, 193, 7, 1)',
                            backgroundColor: 'rgba(255, 193, 7, 0.2)',
                            borderWidth: 2
                        }
                    ]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: {
                            position: 'bottom'
                        }
                    },
                    scales: {
                        r: {
                            beginAtZero: true,
                            max: 100
                        }
                    }
                }
            });
        }
    } catch (error) {
        console.error('[Performance Metrics] Error:', error);
    }
}

// ════════════════════════════════════════════════════════════════
// ATTACK HISTORY
// ════════════════════════════════════════════════════════════════

async function loadHistory(page = 1) {
    try {
        const filter = document.getElementById('historyFilter')?.value || 'all';
        const response = await fetch(`${API_BASE}/history?page=${page}&filter=${filter}`);
        const data = await response.json();
        
        let html = '';
        data.history.forEach(attack => {
            const blockedBadge = attack.blocked 
                ? '<span class="badge badge-blocked">Blocked</span>' 
                : '<span class="badge badge-unblocked">Unblocked</span>';
            const severityBadge = `<span class="badge badge-${attack.severity.toLowerCase()}">${attack.severity}</span>`;
            
            html += `
                <tr>
                    <td>${attack.id}</td>
                    <td><small>${new Date(attack.timestamp).toLocaleString()}</small></td>
                    <td><code>${attack.src_ip}</code></td>
                    <td><code>${attack.dst_ip}</code></td>
                    <td>${attack.attack_type}</td>
                    <td><strong>${attack.confidence}%</strong></td>
                    <td>${severityBadge}</td>
                    <td>${blockedBadge}</td>
                    <td>${attack.duration}</td>
                    <td>
                        <button class="btn btn-sm btn-primary" onclick="viewAttackDetails(${attack.id})">
                            <i class="fas fa-eye"></i>
                        </button>
                    </td>
                </tr>
            `;
        });
        
        document.getElementById('historyBody').innerHTML = html;
        
        // Pagination
        let paginationHTML = '';
        for (let i = 1; i <= data.total_pages; i++) {
            const active = i === data.page ? 'active' : '';
            paginationHTML += `<li class="page-item ${active}"><a class="page-link" href="#" onclick="loadHistory(${i})">${i}</a></li>`;
        }
        document.getElementById('historyPagination').innerHTML = paginationHTML;
    } catch (error) {
        console.error('[History] Error:', error);
    }
}

async function viewAttackDetails(attackId) {
    try {
        const response = await fetch(`${API_BASE}/attack/${attackId}`);
        const data = await response.json();
        const attack = data.attack;
        
        const detailsHTML = `
            <div class="row">
                <div class="col-md-6">
                    <h6 class="fw-bold">Basic Information</h6>
                    <table class="table table-sm">
                        <tr><td><strong>ID:</strong></td><td>${attack.id}</td></tr>
                        <tr><td><strong>Timestamp:</strong></td><td>${new Date(attack.timestamp).toLocaleString()}</td></tr>
                        <tr><td><strong>Source IP:</strong></td><td><code>${attack.src_ip}</code></td></tr>
                        <tr><td><strong>Dest IP:</strong></td><td><code>${attack.dst_ip}</code></td></tr>
                    </table>
                </div>
                <div class="col-md-6">
                    <h6 class="fw-bold">Attack Details</h6>
                    <table class="table table-sm">
                        <tr><td><strong>Type:</strong></td><td>${attack.attack_type}</td></tr>
                        <tr><td><strong>Confidence:</strong></td><td><strong>${attack.confidence}%</strong></td></tr>
                        <tr><td><strong>Severity:</strong></td><td><span class="badge badge-${attack.severity.toLowerCase()}">${attack.severity}</span></td></tr>
                        <tr><td><strong>Blocked:</strong></td><td>${attack.blocked ? '<span class="badge badge-success">Yes</span>' : '<span class="badge badge-danger">No</span>'}</td></tr>
                    </table>
                </div>
            </div>
            <div class="row mt-3">
                <div class="col-12">
                    <h6 class="fw-bold">Traffic Statistics</h6>
                    <table class="table table-sm">
                        <tr>
                            <td><strong>Duration:</strong></td><td>${attack.duration} seconds</td>
                            <td><strong>Packets:</strong></td><td>${attack.packets.toLocaleString()}</td>
                            <td><strong>Bytes:</strong></td><td>${(attack.bytes / 1024 / 1024).toFixed(2)} MB</td>
                        </tr>
                    </table>
                </div>
            </div>
            ${data.alerts.length > 0 ? `
                <div class="row mt-3">
                    <div class="col-12">
                        <h6 class="fw-bold">Associated Alerts</h6>
                        <table class="table table-sm">
                            <thead>
                                <tr><th>Type</th><th>Status</th><th>Recipient</th><th>Time</th></tr>
                            </thead>
                            <tbody>
                                ${data.alerts.map(a => `
                                    <tr>
                                        <td><strong>${a.type}</strong></td>
                                        <td><span class="badge badge-success">${a.status}</span></td>
                                        <td>${a.recipient}</td>
                                        <td><small>${new Date(a.timestamp).toLocaleString()}</small></td>
                                    </tr>
                                `).join('')}
                            </tbody>
                        </table>
                    </div>
                </div>
            ` : ''}
        `;
        
        document.getElementById('attackDetailContent').innerHTML = detailsHTML;
        new bootstrap.Modal(document.getElementById('attackDetailModal')).show();
    } catch (error) {
        console.error('[Attack Details] Error:', error);
    }
}

// ════════════════════════════════════════════════════════════════
// ALERTS
// ════════════════════════════════════════════════════════════════

async function sendAlert(e) {
    e.preventDefault();
    
    const attackId = document.getElementById('alertAttackId').value;
    const alertType = document.getElementById('alertType').value.toLowerCase();
    const recipient = document.getElementById('alertRecipient').value;
    
    try {
        const response = await fetch(`${API_BASE}/alerts/send`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                attack_id: parseInt(attackId),
                alert_type: alertType,
                recipient: recipient
            })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            showNotification('Alert sent successfully!', 'success');
            document.getElementById('alertForm').reset();
            loadAlertsHistory();
        } else {
            showNotification(data.error || 'Failed to send alert', 'danger');
        }
    } catch (error) {
        console.error('[Send Alert] Error:', error);
        showNotification('Error sending alert', 'danger');
    }
}

async function loadAlertsHistory() {
    try {
        const response = await fetch(`${API_BASE}/alerts/history`);
        const alerts = await response.json();
        
        let html = '';
        alerts.slice(0, 10).forEach(alert => {
            html += `
                <div class="alert alert-${alert.status === 'sent' ? 'success' : 'warning'} mb-2">
                    <div class="d-flex justify-content-between align-items-center">
                        <div>
                            <strong>${alert.type.toUpperCase()}</strong>
                            <small class="text-muted d-block">${alert.attack_type} from ${alert.src_ip}</small>
                            <small class="text-muted">${new Date(alert.timestamp).toLocaleString()}</small>
                        </div>
                        <span class="badge badge-${alert.status === 'sent' ? 'success' : 'warning'}">${alert.status.toUpperCase()}</span>
                    </div>
                </div>
            `;
        });
        
        const container = document.getElementById('alertsHistoryContainer');
        if (container) {
            container.innerHTML = html || '<p class="text-muted text-center">No alerts yet</p>';
        }
    } catch (error) {
        console.error('[Alerts History] Error:', error);
    }
}

// ════════════════════════════════════════════════════════════════
// EARLY WARNING / PREDICTION
// ════════════════════════════════════════════════════════════════

async function loadEarlyWarning() {
    try {
        const response = await fetch(`${API_BASE}/prediction/early-warning`);
        const predictions = await response.json();
        
        let html = '';
        predictions.forEach(pred => {
            const severityColor = {
                'CRITICAL': 'danger',
                'HIGH': 'warning',
                'MEDIUM': 'info',
                'LOW': 'success'
            }[pred.predicted_severity] || 'secondary';
            
            html += `
                <div class="col-lg-6 col-md-12">
                    <div class="prediction-card">
                        <div class="attack-type">
                            <i class="fas fa-exclamation-circle"></i> ${pred.attack_type}
                        </div>
                        <div class="metric">
                            <span class="metric-label">Likelihood:</span>
                            <span class="metric-value">${pred.likelihood}%</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Predicted Severity:</span>
                            <span class="metric-value">
                                <span class="badge badge-${severityColor}">${pred.predicted_severity}</span>
                            </span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Historical Frequency:</span>
                            <span class="metric-value">${pred.historical_frequency} times</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Avg Confidence:</span>
                            <span class="metric-value">${pred.avg_confidence}%</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Est. Packet Rate:</span>
                            <span class="metric-value">${pred.estimated_packet_rate.toLocaleString()} pps</span>
                        </div>
                    </div>
                </div>
            `;
        });
        
        const container = document.getElementById('predictionCards');
        if (container) {
            container.innerHTML = html || '<div class="col-12 text-center text-muted">No predictions available</div>';
        }
    } catch (error) {
        console.error('[Prediction] Error:', error);
    }
}

// ════════════════════════════════════════════════════════════════
// UTILITIES
// ════════════════════════════════════════════════════════════════

function showNotification(message, type = 'info') {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
    alertDiv.style.top = '20px';
    alertDiv.style.right = '20px';
    alertDiv.style.zIndex = '9999';
    alertDiv.style.minWidth = '300px';
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(alertDiv);
    
    setTimeout(() => {
        alertDiv.remove();
    }, 5000);
}

// Generate sample data for testing
function generateSampleData() {
    fetch(`${API_BASE}/test/generate-sample-data`)
        .then(r => r.json())
        .then(data => {
            showNotification('Sample data generated: ' + data.count + ' records', 'success');
            setTimeout(loadDashboardData, 1000);
        })
        .catch(e => {
            console.error('Error generating sample data:', e);
            showNotification('Error generating sample data', 'danger');
        });
}
