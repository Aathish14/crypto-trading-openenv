document.addEventListener('DOMContentLoaded', () => {
    const statusBadge = document.getElementById('status-badge');
    const healthStats = document.getElementById('health-stats');
    const envList = document.getElementById('env-list');
    const startBtn = document.getElementById('start-btn');

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
            
            // Show success toast or alert
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
