const hourDropDown = document.getElementById("hour");
const dayDropDown = document.getElementById("day")
let weekdays = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];

for (let i = 0; i < 24; i++) {
    const el = document.createElement("option");
    let txt = ""
    if (i === 0) {
        txt = "12 Am to 12:59 Am";
    } else if (i < 12) {
        txt = String(i) + " Am to " + String(i) + ":59 Am";
    } else if (i === 12) {
        txt = "12 Pm to 12:59 Pm";
    } else {
        txt = String(i - 12) + " Pm to " + String(i - 12) + ":59 Pm";
    }
    el.textContent = txt;
    el.value = String(i);
    hourDropDown.appendChild(el);
}

for (let i = 0; i < weekdays.length; i++) {
    const opt = weekdays[i];
    const el = document.createElement("option");
    el.textContent = opt;
    el.value = String(i);
    dayDropDown.appendChild(el);
}

const tempSlider = document.getElementById("tempSlider");
const tempVal = document.getElementById("tempVal");
tempVal.innerHTML = tempSlider.value;
tempSlider.oninput = function () {
    tempVal.innerHTML = this.value;
}

const windChillSlider = document.getElementById("windChillSlider");
const windChillVal = document.getElementById("windChillVal");
windChillVal.innerHTML = windChillSlider.value;
windChillSlider.oninput = function () {
    windChillVal.innerHTML = this.value;
}

const windSpdSlider = document.getElementById("windSpdSlider");
const windSpdVal = document.getElementById("windSpdVal");
windSpdVal.innerHTML = windSpdSlider.value;
windSpdSlider.oninput = function () {
    windSpdVal.innerHTML = this.value;
}

const humidSlider = document.getElementById("humidSlider");
const humidVal = document.getElementById("humidVal");
humidVal.innerHTML = humidSlider.value;
humidSlider.oninput = function () {
    humidVal.innerHTML = this.value;
}

const pressureSlider = document.getElementById("pressureSlider");
const pressureVal = document.getElementById("pressureVal");
pressureVal.innerHTML = pressureSlider.value;
pressureSlider.oninput = function () {
    pressureVal.innerHTML = this.value;
}

const visibilitySlider = document.getElementById("visibilitySlider");
const visibilityVal = document.getElementById("visibilityVal");
visibilityVal.innerHTML = visibilitySlider.value;
visibilitySlider.oninput = function () {
    visibilityVal.innerHTML = this.value;
}
