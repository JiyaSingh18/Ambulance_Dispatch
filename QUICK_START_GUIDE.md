# 🚀 Quick Start Guide - Enhanced Ambulance Dispatch System

Welcome to the upgraded Emergency Ambulance Dispatch Optimization System! This guide will help you explore all the new features.

---

## 🎯 Getting Started

### Running the Application

```bash
streamlit run ui.py
```

The application will open in your default web browser at `http://localhost:8501`

---

## 📱 Navigation Overview

### Main Tabs

1. **🏠 Dashboard** - Your command center
2. **🗺️ Live Map** - Real-time simulation view
3. **📊 Analytics** - Performance trends and insights
4. **⚡ Performance** - Algorithm comparison tools
5. **📋 Timeline** - Event log and history
6. **🚦 Traffic** - Traffic analysis and control
7. **🎬 Scenarios** - Pre-built and custom scenarios

---

## 🎮 Key Features Guide

### 1. Dashboard Tab (🏠)

**What to do here:**
- Quick start/pause simulation controls
- Monitor overall system health (0-100%)
- View key performance indicators (KPIs)
- Add emergencies and ambulances quickly

**Pro Tips:**
- Watch the "System Health" metric - keep it above 70% for optimal performance
- "Queue Pressure" shows emergency load vs ambulance capacity
- Use quick action buttons for rapid testing

---

### 2. Live Map Tab (🗺️)

**What to do here:**
- Watch ambulances respond to emergencies in real-time
- Toggle path visibility with "Show/Hide Paths" button
- Add batch emergencies (4 at once)
- Place emergencies manually at specific coordinates
- Export dispatch reports as JSON

**Understanding the Map:**
- 🟦 Blue markers = Ambulances
- 🟥 Red markers = Active emergencies (number shows urgency 1-5)
- 🟩 Green markers = Resolved emergencies
- 🟨 Yellow routes = Highlighted ambulance path
- Blue routes = Regular ambulance paths

**Pro Tips:**
- Use the focus unit selector (sidebar) to track specific ambulances
- Enable manual emergency placement for precise testing
- Download reports for offline analysis

---

### 3. Analytics Tab (📊)

**What to do here:**
- Track response time trends over time
- Monitor total distance traveled
- View active emergency patterns
- Analyze optimization performance

**Key Metrics:**
- **Response Time**: Lower is better (target: < 5 minutes)
- **Distance**: Total fleet travel distance
- **Active Emergencies**: Concurrent call load
- **Optimization Cost**: Assignment efficiency

**Pro Tips:**
- Look for the average response time line on charts
- Check peak concurrent emergencies to assess system capacity
- Monitor optimization cost stability

---

### 4. Performance Tab (⚡)

**What to do here:**
- Compare Dijkstra vs A* algorithms
- View execution time benchmarks
- Analyze assignment costs
- Track optimization trends

**Understanding Performance:**
- **Execution Time**: How fast the algorithm runs (lower is better)
- **Assignment Cost**: Quality of ambulance-emergency pairing (lower is better)
- **Performance Trend**: Consistency over time

**Pro Tips:**
- Switch algorithms in sidebar to compare
- A* is typically faster but Dijkstra guarantees optimality
- Hungarian method gives optimal assignments, Random is baseline

---

### 5. Timeline Tab (📋)

**What to do here:**
- Review chronological event log
- See emergency creation and resolution
- Track system changes
- Analyze event patterns

**Event Severity:**
- 🔴 Critical - Urgency 4-5 emergencies, system alerts
- 🟡 Warning - Urgency 3 emergencies, traffic changes
- 🟢 Info/Success - General events, resolutions

**Pro Tips:**
- Use event summary charts to identify patterns
- Check severity distribution for system stress
- Export timeline for post-simulation analysis

---

### 6. Traffic Tab (🚦)

**What to do here:**
- Visualize traffic density heatmap
- Adjust traffic conditions dynamically
- View edge weight statistics
- Analyze traffic distribution

**Traffic Controls:**
- **Increase Traffic** - Adds 50% congestion
- **Normal Traffic** - Resets to baseline
- **Reduce Traffic** - Reduces by 50%

**Pro Tips:**
- Enable "Show Traffic Heatmap" in sidebar first
- Red areas indicate high congestion
- Use traffic scenarios to test resilience
- Check weight distribution histogram for network analysis

---

### 7. Scenarios Tab (🎬)

**What to do here:**
- Launch preset scenarios with one click
- Build custom scenarios with sliders
- Export current state for replay
- Test edge cases and stress scenarios

**Preset Scenarios:**
- 🏙️ **Urban Rush Hour** - High traffic, 8 emergencies (urgency 3-5)
- 🌃 **Quiet Night** - Low traffic, 2 emergencies (urgency 1-3)
- ⚠️ **Mass Casualty** - 15 critical emergencies (urgency 5)

**Custom Scenario Builder:**
- Adjust number of emergencies (1-20)
- Set urgency range (min-max 1-5)
- Add extra ambulances (0-10)
- Control traffic multiplier (0.1x - 3.0x)

**Pro Tips:**
- Start with presets to understand patterns
- Use custom builder for specific test cases
- Export scenarios before major changes
- Test algorithm changes with identical scenarios

---

## 🎛️ Sidebar Controls

### Algorithm Configuration
- **Routing Algorithm**: Choose Dijkstra or A*
- **Assignment Strategy**: Hungarian (optimal) or Random (baseline)
- **Playback Speed**: Real-time or Fast-forward

### Display Options
- **Manual Map Zoom**: Override auto-zoom
- **Show Traffic Heatmap**: Toggle traffic overlay
- **Show Notifications**: Enable/disable alerts
- **Focus on Unit**: Track specific ambulance

### Traffic Scenarios
- **Balanced**: Normal traffic conditions
- **Rush Hour**: 2x traffic congestion
- **Quiet**: 0.3x traffic (light)
- **Custom**: Manual adjustments

---

## 💡 Best Practices

### For Testing
1. Start with a preset scenario
2. Monitor dashboard health metrics
3. Switch algorithms to compare performance
4. Export reports for documentation
5. Review timeline for unexpected events

### For Analysis
1. Run simulation for at least 5-10 minutes
2. Check performance tab for algorithm comparison
3. Review analytics charts for trends
4. Export data at key decision points
5. Compare scenarios with different parameters

### For Demonstrations
1. Use **Live Map** tab for visual impact
2. Enable **Show Paths** for clarity
3. Focus on specific ambulances for storytelling
4. Use **Dashboard** to show system health
5. Reference **Performance** tab for technical details

---

## 🔔 Understanding Notifications

### Notification Types (Sidebar)
- **🚨 Critical**: Immediate attention required
  - High urgency emergencies (4-5)
  - System health below 40%
  - Mass casualty scenarios

- **⚠️ Warning**: Monitor closely
  - Medium urgency emergencies (3)
  - Traffic increases
  - Queue pressure rising

- **ℹ️ Info**: General updates
  - Normal emergencies (1-2)
  - Algorithm changes
  - Ambulance completions

---

## 📊 Key Performance Indicators (KPIs)

### Primary Metrics to Monitor

1. **Overall Health (0-100%)**
   - 70-100%: Excellent (🟢)
   - 40-69%: Fair (🟡)
   - 0-39%: Critical (🔴)

2. **Fleet Utilization**
   - Target: 60-80% (efficient but not overwhelmed)
   - < 50%: Under-utilized
   - > 90%: Overloaded

3. **Average Response Time**
   - Target: < 5 minutes
   - Acceptable: 5-10 minutes
   - Critical: > 10 minutes

4. **Queue Pressure (Active/Total)**
   - Healthy: Active < 50% of Total
   - Stressed: Active > 75% of Total
   - Critical: Active ≥ Total

---

## 🎓 Tutorial: Your First Simulation

### Step-by-Step Guide

1. **Launch Application**
   ```bash
   streamlit run ui.py
   ```

2. **Configure Settings** (Sidebar)
   - Select "Mumbai (OSM)" or "Grid (10x10)"
   - Choose "Dijkstra" routing
   - Select "Hungarian" assignment
   - Set "Real-time" speed

3. **Start Scenario** (Dashboard Tab)
   - Click "Add Emergencies (3x)"
   - Click "Start Simulation"

4. **Monitor Live Map** (Live Map Tab)
   - Watch ambulances respond
   - Enable "Show Paths"
   - Check ambulance status table

5. **Review Performance** (Performance Tab)
   - Wait for 2-3 assignments
   - Check execution time
   - Note assignment costs

6. **Compare Algorithms** (Sidebar)
   - Change routing to "A*"
   - Monitor performance differences
   - Review performance charts

7. **Export Results**
   - Download dispatch report (Live Map tab)
   - Save scenario (Scenarios tab)
   - Take screenshots of key metrics

---

## 🆘 Troubleshooting

### Common Issues

**Issue**: Map not displaying
- **Solution**: Check internet connection (OSM requires network)
- **Alternative**: Switch to Grid mode in sidebar

**Issue**: Simulation too slow
- **Solution**: Change playback to "Fast-forward"
- **Alternative**: Reduce number of ambulances

**Issue**: No performance data
- **Solution**: Run simulation longer (need assignments first)
- **Alternative**: Add more emergencies to trigger assignments

**Issue**: Traffic heatmap not showing
- **Solution**: Enable "Show Traffic Heatmap" in sidebar
- **Check**: Ensure map has loaded completely

---

## 🎯 Advanced Usage

### Benchmarking Algorithms

1. Load identical scenario (use Scenarios tab)
2. Run with Dijkstra for 5 minutes
3. Export report and note metrics
4. Reset simulation
5. Load same scenario
6. Switch to A*
7. Run for 5 minutes
8. Compare reports and Performance tab

### Stress Testing

1. Use "Mass Casualty Event" scenario
2. Monitor system health
3. Add ambulances if health drops < 40%
4. Increase traffic to 2x or 3x
5. Track response time degradation
6. Document maximum capacity

### Custom Research Scenarios

1. Define hypothesis (e.g., "A* faster in high traffic")
2. Build custom scenario with specific parameters
3. Control variables (fix ambulance count, etc.)
4. Run multiple trials
5. Export data for each trial
6. Analyze results in Performance and Analytics tabs

---

## 📞 Support & Resources

### Documentation
- **UI_ENHANCEMENTS.md** - Full feature list
- **QUICK_START_GUIDE.md** - This guide
- **Help & Info** - Sidebar expandable section

### Tips
- Hover over metrics for additional info
- Use tooltips (help icons) for guidance
- Check Timeline tab if something unexpected happens
- Export reports regularly for analysis

---

## 🎉 Have Fun!

The system is designed to be intuitive and informative. Don't be afraid to:
- Try different scenarios
- Switch algorithms mid-simulation
- Test extreme conditions
- Export and analyze data
- Share your findings!

**Happy Dispatching! 🚑**

---

**Version**: 2.0  
**Last Updated**: November 26, 2025  
**Need Help?** Check the sidebar Help & Info section

