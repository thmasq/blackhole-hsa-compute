use anyhow::Result;
use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

pub struct WindowState {
    pub window: Option<Arc<Window>>,
    pub size: winit::dpi::PhysicalSize<u32>,
    pub scale_factor: f64,
}

impl WindowState {
    pub fn new() -> Self {
        Self {
            window: None,
            size: winit::dpi::PhysicalSize::new(800, 600),
            scale_factor: 1.0,
        }
    }

    pub fn create_window(&mut self, event_loop: &ActiveEventLoop) -> Result<Arc<Window>> {
        let window_attributes = Window::default_attributes()
            .with_title("Black Hole Ray Tracer")
            .with_inner_size(self.size)
            .with_min_inner_size(winit::dpi::LogicalSize::new(320, 240));

        let window = Arc::new(event_loop.create_window(window_attributes)?);
        self.window = Some(window.clone());

        // Get actual size and scale
        self.size = window.inner_size();
        self.scale_factor = window.scale_factor();

        log::info!(
            "Created window: {:?} at {}x scale",
            self.size,
            self.scale_factor
        );

        Ok(window)
    }
}

pub struct AppState {
    pub window_state: WindowState,
    pub should_quit: bool,
    pub renderer: Option<crate::renderer::Renderer>,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            window_state: WindowState::new(),
            should_quit: false,
            renderer: None,
        }
    }
}

impl ApplicationHandler for AppState {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window_state.window.is_none() {
            match self.window_state.create_window(event_loop) {
                Ok(window) => {
                    // Initialize renderer with the window
                    match pollster::block_on(crate::renderer::Renderer::new(
                        window.clone(),
                        self.window_state.size.width,
                        self.window_state.size.height,
                    )) {
                        Ok(renderer) => {
                            self.renderer = Some(renderer);
                            log::info!("Renderer initialized");
                        }
                        Err(e) => {
                            log::error!("Failed to initialize renderer: {}", e);
                            self.should_quit = true;
                        }
                    }
                }
                Err(e) => {
                    log::error!("Failed to create window: {}", e);
                    self.should_quit = true;
                }
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                log::info!("Close requested");
                self.should_quit = true;
                event_loop.exit();
            }

            WindowEvent::Resized(physical_size) => {
                if physical_size.width > 0 && physical_size.height > 0 {
                    self.window_state.size = physical_size;
                    if let Some(renderer) = &mut self.renderer {
                        renderer.resize(physical_size.width, physical_size.height);
                    }
                }
            }

            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                self.window_state.scale_factor = scale_factor;
                log::info!("Scale factor changed to: {}", scale_factor);
            }

            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(keycode),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => match keycode {
                KeyCode::Escape | KeyCode::KeyQ => {
                    log::info!("Quit key pressed");
                    self.should_quit = true;
                    event_loop.exit();
                }
                KeyCode::Space => {
                    if let Some(renderer) = &mut self.renderer {
                        renderer.toggle_animation();
                    }
                }
                _ => {}
            },

            WindowEvent::RedrawRequested => {
                if let Some(renderer) = &mut self.renderer {
                    match renderer.render() {
                        Ok(_) => {}
                        Err(e) => {
                            log::error!("Render error: {}", e);
                        }
                    }
                    // Request next frame
                    if let Some(window) = &self.window_state.window {
                        window.request_redraw();
                    }
                }
            }

            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        // Request redraw for continuous rendering
        if let Some(window) = &self.window_state.window {
            window.request_redraw();
        }
    }
}

pub fn create_event_loop() -> Result<EventLoop<()>> {
    let event_loop = EventLoop::builder().build()?;

    // Force Wayland if available
    if cfg!(target_os = "linux") {
        std::env::set_var("WINIT_UNIX_BACKEND", "wayland");
    }

    Ok(event_loop)
}
