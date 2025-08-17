use anyhow::Result;
use std::sync::Arc;
use winit::window::Window;

pub struct Renderer {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: (u32, u32),
    texture: Option<wgpu::Texture>,
    texture_view: Option<wgpu::TextureView>,
    compute: Option<crate::ffi::BlackholeCompute>,
    frame_buffer: crate::frame_buffer::SharedFrameBuffer,
    render_pipeline: wgpu::RenderPipeline,
    bind_group: Option<wgpu::BindGroup>,
    sampler: wgpu::Sampler,
    bind_group_layout: wgpu::BindGroupLayout,
    animation_time: f32,
    animate: bool,
}

impl Renderer {
    pub async fn new(window: Arc<Window>, width: u32, height: u32) -> Result<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let surface = instance.create_surface(window.clone())?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| anyhow::anyhow!("Failed to find suitable adapter"))?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Renderer Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await?;

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width,
            height,
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Texture Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // Create render pipeline
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Display Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/display.wgsl").into()),
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        // Create sampler
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Initialize compute library
        let compute = match crate::ffi::BlackholeCompute::new(width, height, 60000) {
            Ok(c) => Some(c),
            Err(e) => {
                log::error!(
                    "Failed to initialize compute: {}. Will show test pattern.",
                    e
                );
                None
            }
        };

        let frame_buffer = crate::frame_buffer::SharedFrameBuffer::new(width, height);

        Ok(Self {
            surface,
            device,
            queue,
            config,
            size: (width, height),
            texture: None,
            texture_view: None,
            compute,
            frame_buffer,
            render_pipeline,
            bind_group: None,
            sampler,
            bind_group_layout,
            animation_time: 0.0,
            animate: true,
        })
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            log::info!("Resizing to {}x{}", width, height);
            self.size = (width, height);
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);

            // Note: We'd need to reinitialize the compute library for proper resize
            // For now, we'll keep the same resolution and stretch
        }
    }

    pub fn toggle_animation(&mut self) {
        self.animate = !self.animate;
        log::info!("Animation: {}", if self.animate { "ON" } else { "OFF" });
    }

    fn update_frame(&mut self) -> Result<()> {
        if self.animate {
            self.animation_time += 0.016; // ~60 FPS
        }

        if let Some(compute) = &mut self.compute {
            // Orbit camera
            let azimuth = self.animation_time * 0.5;
            let elevation = 1.5 + (self.animation_time * 0.3).sin() * 0.3;
            let radius = 6.34194e10;

            compute.update_camera(azimuth, elevation, radius);

            // Render frame (borrow ends after this block)
            let frame_data = {
                let data = compute.render_frame()?;
                // Also update framebuffer while we still hold the reference
                self.frame_buffer.update(data)?;
                data.to_vec() // clone into owned Vec<u8>
            };

            // Now we can use self again safely
            self.update_texture(&frame_data);
        } else {
            // Generate test pattern
            self.generate_test_pattern();
        }

        Ok(())
    }

    fn generate_test_pattern(&mut self) {
        let (width, height) = self
            .compute
            .as_ref()
            .map(|c| c.dimensions())
            .unwrap_or((800, 600));

        let mut data = vec![0u8; (width * height * 4) as usize];
        let time = self.animation_time;

        for y in 0..height {
            for x in 0..width {
                let idx = ((y * width + x) * 4) as usize;
                let fx = x as f32 / width as f32;
                let fy = y as f32 / height as f32;

                // Animated gradient
                data[idx] = ((fx * 255.0 * (1.0 + time.sin())) as u8).min(255);
                data[idx + 1] = ((fy * 255.0 * (1.0 + time.cos())) as u8).min(255);
                data[idx + 2] = (((fx + fy) * 127.0 * (1.0 + (time * 2.0).sin())) as u8).min(255);
                data[idx + 3] = 255;
            }
        }

        self.frame_buffer.update(&data).ok();
        self.update_texture(&data);
    }

    fn update_texture(&mut self, data: &[u8]) {
        let (width, height) = self
            .compute
            .as_ref()
            .map(|c| c.dimensions())
            .unwrap_or((800, 600));

        // Create or recreate texture if needed
        if self.texture.is_none() {
            let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Frame Texture"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });

            let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

            // Create bind group
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Texture Bind Group"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            });

            self.texture = Some(texture);
            self.texture_view = Some(texture_view);
            self.bind_group = Some(bind_group);
        }

        // Write texture data
        if let Some(texture) = &self.texture {
            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(width * 4),
                    rows_per_image: Some(height),
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );
        }
    }

    pub fn render(&mut self) -> Result<()> {
        // Update frame data
        self.update_frame()?;

        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.1,
                            b: 0.1,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            if let Some(bind_group) = &self.bind_group {
                render_pass.set_pipeline(&self.render_pipeline);
                render_pass.set_bind_group(0, bind_group, &[]);
                render_pass.draw(0..6, 0..1);
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}
