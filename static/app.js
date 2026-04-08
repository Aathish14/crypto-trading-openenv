document.addEventListener('DOMContentLoaded', () => {
    const statusBadge = document.getElementById('status-badge');
    const healthStats = document.getElementById('health-stats');
    const envList = document.getElementById('env-list');
    const startBtn = document.getElementById('start-btn');

    // Start Button Interaction
    startBtn.addEventListener('click', () => {
        startBtn.innerHTML = '<span class="btn-icon">⏳</span> Starting...';
        startBtn.style.opacity = '0.7';
        startBtn.disabled = true;
        
        // Mock simulation start
        setTimeout(() => {
            alert('Simulation start logic will be implemented in the next phase!');
            startBtn.innerHTML = '<span class="btn-icon">▶</span> Start Simulation';
            startBtn.style.opacity = '1';
            startBtn.disabled = false;
        }, 1500);
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
