"use strict";

// The globe adapts the MIT-licensed Three.js treatment from
// https://github.com/abhijandyala/3D-Earth.
(function launchAegisIntro() {
  const intro = document.getElementById("aegis-intro");
  const earthHost = document.getElementById("intro-earth");
  const canvas = document.getElementById("intro-earth-canvas");
  const start = document.getElementById("intro-start");
  const more = document.getElementById("intro-more");
  const info = document.getElementById("intro-info");
  const infoClose = document.getElementById("intro-info-close");
  if (!intro || !earthHost || !canvas || !start || !more || !info || !infoClose) return;

  let finished = false;
  let launching = false;
  let stopGlobe = () => {};

  function finishIntro() {
    if (finished) return;
    finished = true;
    stopGlobe();
    intro.remove();
    document.body.classList.remove("intro-active");
    window.dispatchEvent(new Event("resize"));
    window.dispatchEvent(new CustomEvent("aegis:intro-complete"));
  }

  function setInfoOpen(open) {
    if (launching || finished) return;
    intro.classList.toggle("info-open", open);
    info.setAttribute("aria-hidden", String(!open));
    more.setAttribute("aria-expanded", String(open));
    if (open) infoClose.focus();
    else more.focus();
  }

  function startIntro() {
    if (finished || launching) return;
    launching = true;
    intro.classList.remove("info-open");
    info.setAttribute("aria-hidden", "true");
    more.setAttribute("aria-expanded", "false");
    intro.classList.add("is-launching");
    document.body.classList.remove("intro-active");
    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    window.setTimeout(finishIntro, reducedMotion ? 1150 : 4000);
  }

  start.addEventListener("click", startIntro);
  more.addEventListener("click", () => setInfoOpen(true));
  infoClose.addEventListener("click", () => setInfoOpen(false));
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !finished) {
      event.preventDefault();
      event.stopImmediatePropagation();
      if (intro.classList.contains("info-open")) setInfoOpen(false);
    }
  }, true);

  intro.querySelector(".intro-door-right").addEventListener("animationend", (event) => {
    if (event.animationName === "intro-door-right") finishIntro();
  });

  intro.classList.add("is-ready");

  setupGlobe(canvas, earthHost)
    .then((cleanup) => {
      if (finished) cleanup();
      else stopGlobe = cleanup;
    })
    .catch((error) => {
      // The textured CSS sphere remains visible if WebGL or the CDN is unavailable.
      console.warn("Aegis intro globe fell back to CSS rendering:", error);
    });
})();

async function setupGlobe(canvas, host) {
  const THREE = await import("https://cdn.jsdelivr.net/npm/three@0.161.0/build/three.module.js");

  const renderer = new THREE.WebGLRenderer({
    canvas,
    alpha: true,
    antialias: true,
    powerPreference: "high-performance",
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setClearColor(0x000000, 0);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.08;

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 100);
  camera.position.set(0, 0, 4.3);

  const earthGroup = new THREE.Group();
  earthGroup.position.y = -1.95;
  earthGroup.rotation.z = THREE.MathUtils.degToRad(-23.4);
  earthGroup.rotation.x = THREE.MathUtils.degToRad(4);
  scene.add(earthGroup);

  // Layered deterministic point fields adapt the clean star treatment from
  // the referenced 3D-Earth project without adding a heavy sky texture.
  const fineStars = createStarfield(THREE, 1800, 0.021, 0.72, 1907);
  const brightStars = createStarfield(THREE, 95, 0.052, 0.95, 2411);
  scene.add(fineStars, brightStars);

  const loader = new THREE.TextureLoader();
  const [earthTexture, lightsTexture, cloudsTexture] = await Promise.all([
    loader.loadAsync("assets/earth/earthmap.jpg"),
    loader.loadAsync("assets/earth/earth_lights.png"),
    loader.loadAsync("assets/earth/cloud_combined.jpg"),
  ]);
  earthTexture.colorSpace = THREE.SRGBColorSpace;
  lightsTexture.colorSpace = THREE.SRGBColorSpace;
  cloudsTexture.colorSpace = THREE.SRGBColorSpace;

  const geometry = new THREE.SphereGeometry(2.08, 128, 128);
  const earth = new THREE.Mesh(
    geometry,
    new THREE.MeshPhongMaterial({
      map: earthTexture,
      shininess: 7,
      specular: new THREE.Color(0x173b46),
    }),
  );
  earth.rotation.y = THREE.MathUtils.degToRad(-74);
  earthGroup.add(earth);

  const cityLights = new THREE.Mesh(
    geometry,
    new THREE.MeshBasicMaterial({
      map: lightsTexture,
      transparent: true,
      opacity: 0.48,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    }),
  );
  cityLights.rotation.y = earth.rotation.y;
  cityLights.scale.setScalar(1.001);
  earthGroup.add(cityLights);

  const clouds = new THREE.Mesh(
    geometry,
    new THREE.MeshStandardMaterial({
      map: cloudsTexture,
      alphaMap: cloudsTexture,
      color: 0xffffff,
      transparent: true,
      opacity: 0.38,
      alphaTest: 0.015,
      blending: THREE.NormalBlending,
      depthWrite: false,
    }),
  );
  clouds.rotation.y = earth.rotation.y;
  clouds.scale.setScalar(1.008);
  earthGroup.add(clouds);

  const glareTexture = makeGlareTexture(THREE);
  const glareMaterial = new THREE.SpriteMaterial({
    map: glareTexture,
    color: 0xd8f8ff,
    transparent: true,
    opacity: 0.92,
    blending: THREE.AdditiveBlending,
    depthTest: false,
    depthWrite: false,
  });
  const sunGlare = new THREE.Sprite(glareMaterial);
  sunGlare.position.set(-0.8, -0.15, 1);
  sunGlare.scale.set(1.5, 1.5, 1);
  sunGlare.renderOrder = 10;
  scene.add(sunGlare);

  const streakMaterial = glareMaterial.clone();
  streakMaterial.opacity = 0.45;
  const glareStreak = new THREE.Sprite(streakMaterial);
  glareStreak.position.copy(sunGlare.position);
  glareStreak.scale.set(4.4, 0.11, 1);
  glareStreak.renderOrder = 11;
  scene.add(glareStreak);

  const keyLight = new THREE.DirectionalLight(0xf1fdff, 3.2);
  keyLight.position.set(-3.7, 2.2, 4.5);
  scene.add(keyLight);

  const rimLight = new THREE.DirectionalLight(0x3abef9, 1.15);
  rimLight.position.set(4, -1.5, 1);
  scene.add(rimLight);
  scene.add(new THREE.AmbientLight(0x20343d, 0.52));

  function resize() {
    const rect = host.getBoundingClientRect();
    const width = Math.max(1, Math.round(rect.width));
    const height = Math.max(1, Math.round(rect.height));
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  }

  resize();
  const resizeObserver = new ResizeObserver(resize);
  resizeObserver.observe(host);

  let animationFrame = 0;
  let active = true;
  let previousTime = performance.now();

  function render(time) {
    if (!active) return;
    const delta = Math.min((time - previousTime) / 1000, 0.05);
    previousTime = time;
    earth.rotation.y += delta * 0.14;
    cityLights.rotation.y += delta * 0.14;
    clouds.rotation.y += delta * 0.19;
    fineStars.rotation.y -= delta * 0.0007;
    brightStars.rotation.y -= delta * 0.0004;
    renderer.render(scene, camera);
    animationFrame = requestAnimationFrame(render);
  }

  renderer.render(scene, camera);
  host.classList.add("webgl-ready");
  animationFrame = requestAnimationFrame(render);

  return function cleanupGlobe() {
    active = false;
    cancelAnimationFrame(animationFrame);
    resizeObserver.disconnect();
    earthTexture.dispose();
    lightsTexture.dispose();
    cloudsTexture.dispose();
    geometry.dispose();
    earth.material.dispose();
    cityLights.material.dispose();
    clouds.material.dispose();
    fineStars.geometry.dispose();
    fineStars.material.dispose();
    brightStars.geometry.dispose();
    brightStars.material.dispose();
    glareTexture.dispose();
    glareMaterial.dispose();
    streakMaterial.dispose();
    renderer.dispose();
  };
}

function createStarfield(THREE, count, size, opacity, initialSeed) {
  let seed = initialSeed;
  const random = () => {
    seed = (seed * 16807) % 2147483647;
    return (seed - 1) / 2147483646;
  };

  const positions = new Float32Array(count * 3);
  for (let index = 0; index < count; index += 1) {
    positions[index * 3] = (random() - 0.5) * 15;
    positions[index * 3 + 1] = (random() - 0.5) * 15;
    positions[index * 3 + 2] = -2 - random() * 9;
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  const material = new THREE.PointsMaterial({
    color: 0xe9f6ff,
    size,
    transparent: true,
    opacity,
    sizeAttenuation: true,
    depthWrite: false,
  });
  return new THREE.Points(geometry, material);
}

function makeGlareTexture(THREE) {
  const glareCanvas = document.createElement("canvas");
  glareCanvas.width = 512;
  glareCanvas.height = 512;
  const context = glareCanvas.getContext("2d");
  const gradient = context.createRadialGradient(256, 256, 0, 256, 256, 256);
  gradient.addColorStop(0, "rgba(255,255,255,1)");
  gradient.addColorStop(0.025, "rgba(235,252,255,.98)");
  gradient.addColorStop(0.08, "rgba(134,229,255,.75)");
  gradient.addColorStop(0.24, "rgba(54,173,222,.24)");
  gradient.addColorStop(0.52, "rgba(18,93,145,.08)");
  gradient.addColorStop(1, "rgba(0,0,0,0)");
  context.fillStyle = gradient;
  context.fillRect(0, 0, 512, 512);
  return new THREE.CanvasTexture(glareCanvas);
}
