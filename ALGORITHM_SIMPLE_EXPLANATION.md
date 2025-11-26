# 🚑 Algorithm Explanation - One Slide Summary

---

## **How the Ambulance Dispatch System Works**

### **Step 1: Calculate Travel Costs** 📍
```
Edge Weight = Distance × Traffic
Example: 5 km road with 2× traffic = 10 time units
```

### **Step 2: Find Shortest Paths** 🛣️
**Dijkstra or A*** finds fastest route from each ambulance to each emergency
```
       Emergency 1  Emergency 2  Emergency 3
Amb A      8 min       15 min       12 min
Amb B     12 min        9 min       11 min
Amb C     14 min       10 min        7 min
```

### **Step 3: Optimal Assignment** 🎯
**Hungarian Algorithm** finds best pairing to minimize total response time
```
✓ Amb A → Emergency 1 (8 min)
✓ Amb B → Emergency 2 (9 min)  
✓ Amb C → Emergency 3 (7 min)
─────────────────────────────
  Total: 24 minutes (OPTIMAL)
```

### **Result** ✨
System assigns ambulances optimally, minimizing total emergency response time across the entire city!

---

## **Key Formulas**

**Travel Time:**  
`Cost = Distance × Traffic`

**Total System Cost:**  
`Total = Sum of all assigned ambulance-to-emergency costs`

**Goal:**  
`Minimize Total Cost = Minimize Response Time = Save More Lives`

---

