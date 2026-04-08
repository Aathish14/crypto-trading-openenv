document.addEventListener('DOMContentLoaded', () => {
    const statusBadge = document.getElementById('status-badge');
    const healthStats = document.getElementById('health-stats');
    const envList = document.getElementById('env-list');
    const startBtn = document.getElementById('start-btn');

    // Chart Instances
    let charts = {
        balance: null,
        reward: null,
        action: null
    };

    // Chart.js Default Config
    Chart.defaults.color = '#a1a1aa';
    Chart.defaults.font.family = "'Inter', sans-serif";
    Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.08)';

    function getChartConfig(type, label, data, color) {
        return {
            type: type,
            data: {
                labels: Array.from({length: data.length}, (_, i) => i),
                datasets: [{
                    label: label,
                    data: data,
                    borderColor: color,
                    backgroundColor: type === 'bar' ? color + '44' : 'transparent',
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.4,
                    fill: type === 'bar'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        backgroundColor: '#18181b',
                        titleColor: '#ffffff',
                        bodyColor: '#a1a1aa',
                        borderColor: 'rgba(255, 255, 255, 0.1)',
                        borderWidth: 1
                    }
                },
                scales: {
                    x: { display: type === 'bar', grid: { display: false } },
                    y: { grid: { color: 'rgba(255, 255, 255, 0.04)' } }
                }
            }
        };
    }

    function updateCharts(data) {
        // Clear previous charts
        Object.values(charts).forEach(chart => chart && chart.destroy());

        // Balance Chart
        charts.balance = new Chart(
            document.getElementById('balanceChart'),
            getChartConfig('line', 'Portfolio Balance', data.balance, '#3b82f6')
        );

        // Reward Chart
        charts.reward = new Chart(
            document.getElementById('rewardChart'),
            getChartConfig('line', 'Step Reward', data.rewards, '#8b5cf6')
        );

        // Action Distribution Chart
        const actionCounts = Array(9).fill(0);
        data.actions.forEach(stepActions => {
            stepActions.forEach(a => actionCounts[a]++);
        });
        
        const actionLabels = ['Hold', 'B 25%', 'B 50%', 'B 75%', 'B 100%', 'S 25%', 'S 50%', 'S 75%', 'S 100%'];
        
        charts.action = new Chart(
            document.getElementById('actionChart'),
            {
                type: 'bar',
                data: {
                    labels: actionLabels,
                    datasets: [{
                        label: 'Action Frequency',
                        data: actionCounts,
                        backgroundColor: '#10b981',
                        borderRadius: 6
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        x: { grid: { display: false } },
                        y: { grid: { color: 'rgba(255, 255, 255, 0.04)' } }
                    }
                }
            }
        );
    }

    // Start Button Interaction
    startBtn.addEventListener('click', async () => {
        const originalText = startBtn.innerHTML;
        startBtn.innerHTML = '<span class="btn-icon">⏳</span> Simulating...';
        startBtn.classList.add('loading');
        startBtn.disabled = true;
        
        try {
            const response = await fetch('/v1/simulate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    steps: 100,
                    strategy: 'random'
                })
            });
            
            if (!response.ok) throw new Error('Simulation failed');
            
            const results = await response.json();
            console.log('Simulation Results:', results);
            
            // Update Charts
            updateCharts(results);
            
            // Show success alert
            const lastBalance = results.balance[results.balance.length - 1];
            alert(`Simulation Complete!\nFinal Portfolio Value: $${lastBalance.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`);
            
        } catch (error) {
            console.error('Error starting simulation:', error);
            alert('Error running simulation. Please check the console for details.');
        } finally {
            startBtn.innerHTML = originalText;
            startBtn.classList.remove('loading');
            startBtn.disabled = false;
        }
    });

    // Helper to create stat item
    function createStatItem(label, value) {
        return `
            <div class="stat-item">
                <span class="stat-label">${label}</span>
                <span class="stat-value">${value}</span>
            </div>
        `;
    }

    // Fetch Health Data
    async function fetchHealth() {
        try {
            const response = await fetch('/health');
            const data = await response.json();
            
            statusBadge.className = 'badge online';
            statusBadge.querySelector('.text').textContent = 'Online';
            
            healthStats.innerHTML = '';
            for (const [key, value] of Object.entries(data)) {
                healthStats.innerHTML += createStatItem(key.charAt(0).toUpperCase() + key.slice(1), value);
            }
        } catch (error) {
            console.error('Error fetching health:', error);
            statusBadge.className = 'badge offline';
            statusBadge.querySelector('.text').textContent = 'Offline';
            healthStats.innerHTML = '<p style="color: var(--danger)">Failed to load health status.</p>';
        }
    }

    // Fetch Environments
    async function fetchEnvs() {
        try {
            const response = await fetch('/v1/envs');
            const data = await response.json();
            
            envList.innerHTML = '';
            // Assuming data is an array of env IDs or a collection
            if (Array.isArray(data) && data.length > 0) {
                data.forEach(envId => {
                    envList.innerHTML += createStatItem('Environment ID', envId);
                });
            } else if (typeof data === 'object') {
                // If it's a dictionary of env info
                Object.keys(data).forEach(envId => {
                    envList.innerHTML += createStatItem('ID', envId);
                });
            } else {
                envList.innerHTML = '<p>No environments discovered.</p>';
            }
        } catch (error) {
            console.error('Error fetching envs:', error);
            envList.innerHTML = '<p style="color: var(--danger)">Failed to load environments.</p>';
        }
    }

    // Initial Load
    fetchHealth();
    fetchEnvs();

    // Auto-refresh every 30 seconds
    setInterval(() => {
        fetchHealth();
        fetchEnvs();
    }, 30000);
});
