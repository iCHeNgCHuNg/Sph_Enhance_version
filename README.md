# SPH Channel Flow Simulation

## 1. Project Purpose
本專案以 Smoothed Particle Hydrodynamics (SPH) 方法為基礎，
在原始「星體結構模擬」程式的基礎上，修改並延伸為
**工程流體力學中的管道流（channel flow）模擬**，
展示外加壓力梯度與黏滯效應對流速分佈的影響。

## 2. Physical & Numerical Model
- 使用 SPH 核函數計算粒子間相互作用
- 關閉星體模型中的徑向引力與密度剖面分析
- 引入：
  - 水平壓力梯度作為流動驅動力
  - 黏滯阻尼模擬流體內部摩擦
  - 上下邊界施加 no-slip 條件

## 3. Program Structure
- `sph_enhanced_version.py`：主模擬程式
- 上圖：粒子在管道中的運動情形
- 下圖：穩態後的平均速度剖面 v_x(y)

## 4. Development Process
- 參考 GitHub 上 Philip Mocz 的 SPH 教學程式
- 修改加速度計算方式，改為工程流體驅動模型
- 新增速度剖面統計與即時視覺化
- 使用 ChatGPT 協助理解 SPH 模型與程式修改方向

## 5. References
- Philip Mocz, *Create Your Own Smoothed-Particle-Hydrodynamics Simulation*
- 課堂講義：SPH 與流體力學基礎
