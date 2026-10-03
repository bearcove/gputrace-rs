mod mtsp;

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use plist::Value;
use serde::Serialize;

use crate::error::{Error, Result};

pub use mtsp::{
    CDispatchRecord, CRecord, CiRecord, CiululRecord, CtRecord, CtURecord, CttRecord, CuRecord,
    CuiRecord, CulRecord, CululRecord, CuwRecord, MTLResourceUsage, MTSPHeader, MTSPRecord,
    RecordType, ResourceBinding,
};

pub const MAGIC_MTSP: &[u8; 4] = b"MTSP";

#[derive(Debug, Clone, Serialize)]
pub struct TraceBundle {
    pub path: PathBuf,
    pub metadata: Metadata,
    pub capture_path: PathBuf,
    pub capture_len: u64,
    pub device_resources: Vec<DeviceResource>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct Metadata {
    pub uuid: Option<String>,
    pub capture_version: Option<i64>,
    pub graphics_api: Option<i64>,
    pub device_id: Option<i64>,
    pub native_pointer_size: Option<i64>,
    pub captured_frames_count: Option<i64>,
    pub boundary_less: Option<bool>,
    pub library_link_versions: BTreeMap<String, i64>,
    pub unused_buffer_count: Option<i64>,
    pub unused_texture_count: Option<i64>,
    pub unused_function_count: Option<i64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DeviceResource {
    pub path: PathBuf,
    pub len: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct TraceSummary {
    pub trace_name: String,
    pub uuid: Option<String>,
    pub capture_version: Option<i64>,
    pub graphics_api: Option<i64>,
    pub device_id: Option<i64>,
    pub capture_len: u64,
    pub device_resource_count: usize,
    pub device_resource_bytes: u64,
}

pub type PipelineFunctionMap = BTreeMap<u64, String>;

#[derive(Debug, Clone, Default, Serialize)]
pub struct CommandBuffer {
    pub index: usize,
    pub timestamp: u64,
    pub offset: usize,
    /// 16-char hex artifact hashes carried in the CUUU sub-record's payload —
    /// these correspond to subdirectories of the trace bundle (e.g.,
    /// `F34AD02BAECBD543`) that hold per-CB resource snapshots.
    pub artifact_hashes: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ComputeEncoder {
    pub index: usize,
    pub address: u64,
    pub label: String,
    pub offset: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct DispatchCall {
    pub index: usize,
    pub offset: usize,
    pub encoder_id: Option<u64>,
    pub pipeline_addr: Option<u64>,
    pub kernel_name: Option<String>,
    pub buffers: Vec<BoundBuffer>,
    pub grid_size: [u32; 3],
    pub group_size: [u32; 3],
}

#[derive(Debug, Clone, Serialize)]
pub struct KernelStat {
    pub name: String,
    pub pipeline_addr: u64,
    pub dispatch_count: usize,
    pub encoder_labels: BTreeMap<String, usize>,
    pub buffers: BTreeMap<String, usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CommandBufferRegion {
    pub command_buffer: CommandBuffer,
    pub end_offset: usize,
    pub encoders: Vec<ComputeEncoder>,
    pub pipeline_events: Vec<PipelineStateEvent>,
    pub dispatches: Vec<DispatchCall>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PipelineStateEvent {
    pub offset: usize,
    pub encoder_addr: u64,
    pub pipeline_addr: u64,
    pub function_addr: u64,
    pub buffers: Vec<BoundBuffer>,
}

#[derive(Debug, Clone, Serialize)]
pub struct BoundBuffer {
    pub address: u64,
    pub name: Option<String>,
    pub index: usize,
    pub usage: MTLResourceUsage,
}

#[derive(Debug, Clone, Serialize)]
pub struct BufferAccessStat {
    pub name: String,
    pub address: Option<u64>,
    pub use_count: usize,
    pub dispatch_count: usize,
    pub encoder_count: usize,
    pub command_buffer_count: usize,
    pub first_dispatch_index: usize,
    pub last_dispatch_index: usize,
    pub kernels: BTreeMap<String, usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct BufferLifecycleStat {
    pub name: String,
    pub address: Option<u64>,
    pub first_command_buffer_index: usize,
    pub last_command_buffer_index: usize,
    pub first_dispatch_index: usize,
    pub last_dispatch_index: usize,
    pub command_buffer_span: usize,
    pub dispatch_span: usize,
    pub use_count: usize,
    pub encoder_count: usize,
    pub kernels: BTreeMap<String, usize>,
}

impl TraceBundle {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let info = match fs::metadata(&path) {
            Ok(info) => info,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Err(Error::NotFound(path));
            }
            Err(error) => return Err(error.into()),
        };
        if !info.is_dir() {
            return Err(Error::NotDirectory(path));
        }

        let metadata = parse_metadata(&path)?;
        let capture_path = load_capture_path(&path)?;
        let capture_len = fs::metadata(&capture_path)?.len();
        let resource_path = capture_path.parent().unwrap_or(&path);
        let device_resources = load_device_resources(resource_path)?;

        Ok(Self {
            path,
            metadata,
            capture_path,
            capture_len,
            device_resources,
        })
    }

    pub fn summary(&self) -> TraceSummary {
        TraceSummary {
            trace_name: self
                .path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("unknown")
                .to_owned(),
            uuid: self.metadata.uuid.clone(),
            capture_version: self.metadata.capture_version,
            graphics_api: self.metadata.graphics_api,
            device_id: self.metadata.device_id,
            capture_len: self.capture_len,
            device_resource_count: self.device_resources.len(),
            device_resource_bytes: self.device_resources.iter().map(|entry| entry.len).sum(),
        }
    }

    pub fn capture_data(&self) -> Result<Vec<u8>> {
        Ok(fs::read(&self.capture_path)?)
    }

    pub fn mtsp_header(&self) -> Result<MTSPHeader> {
        let data = self.capture_data()?;
        MTSPHeader::parse(&data)
    }

    pub fn mtsp_records(&self) -> Result<Vec<MTSPRecord>> {
        let data = self.capture_data()?;
        MTSPRecord::parse_stream(&data)
    }

    /// Blobs packed into the bundle's `store0`: a run of zlib streams, each
    /// keyed (in `index`, and by the records that reference it) by the
    /// uppercase hex xxh64 of its decompressed bytes, without zero padding.
    /// Blobs too small for a file of their own live here, among them the
    /// descriptors of specialized functions.
    pub fn store_blobs(&self) -> Result<BTreeMap<String, Vec<u8>>> {
        let path = self.path.join("store0");
        let mut blobs = BTreeMap::new();
        let Ok(data) = fs::read(&path) else {
            return Ok(blobs);
        };
        let mut offset = 0;
        while offset < data.len() {
            let mut decoder = flate2::Decompress::new(true);
            let mut out = Vec::new();
            loop {
                out.reserve(64 * 1024);
                let before_in = decoder.total_in();
                let status = decoder
                    .decompress_vec(&data[offset..], &mut out, flate2::FlushDecompress::None)
                    .map_err(|_| Error::InvalidTrace("store0: bad zlib stream"))?;
                if status == flate2::Status::StreamEnd {
                    break;
                }
                if decoder.total_in() == before_in && out.len() < out.capacity() {
                    return Err(Error::InvalidTrace("store0: truncated zlib stream"));
                }
            }
            offset += decoder.total_in() as usize;
            blobs.insert(format!("{:X}", xxhash_rust::xxh64::xxh64(&out, 0)), out);
        }
        Ok(blobs)
    }

    pub fn pipeline_function_map(&self) -> Result<PipelineFunctionMap> {
        let capture = self.capture_data()?;
        let mut label_map = BTreeMap::new();
        let mut blob_refs = BTreeMap::new();
        collect_labels_from_data(&capture, &mut label_map, &mut blob_refs)?;
        for resource in &self.device_resources {
            let data = fs::read(&resource.path)?;
            collect_labels_from_data(&data, &mut label_map, &mut blob_refs)?;
        }
        // Specialized functions have no CS name record; their CUt record
        // points at a descriptor blob that names them.
        if blob_refs.keys().any(|address| !label_map.contains_key(address)) {
            let blobs = self.store_blobs()?;
            for (address, key) in blob_refs {
                if let Some(name) = blobs.get(&key).and_then(|blob| function_descriptor_name(blob)) {
                    label_map.entry(address).or_insert(name);
                }
            }
        }

        tracing::debug!(labels = label_map.len(), "function labels");
        for (address, label) in &label_map {
            tracing::trace!(address = format_args!("{address:#x}"), %label, "function label");
        }
        let mut result = BTreeMap::new();
        collect_pipeline_mappings_from_data(&capture, &label_map, &mut result)?;
        for resource in &self.device_resources {
            let data = fs::read(&resource.path)?;
            collect_pipeline_mappings_from_data(&data, &label_map, &mut result)?;
        }
        Ok(result)
    }

    pub fn command_buffers(&self) -> Result<Vec<CommandBuffer>> {
        let data = self.capture_data()?;
        Ok(parse_command_buffers(&data))
    }

    pub fn compute_encoders(&self) -> Result<Vec<ComputeEncoder>> {
        let capture = self.capture_data()?;
        let mut encoders = parse_compute_encoders(&capture);
        if encoders.is_empty() {
            for resource in &self.device_resources {
                let data = fs::read(&resource.path)?;
                encoders.extend(parse_compute_encoders(&data));
            }
        }
        dedupe_encoders(encoders)
    }

    pub fn dispatch_calls(&self) -> Result<Vec<DispatchCall>> {
        let capture = self.capture_data()?;
        let records = MTSPRecord::parse_stream(&capture)?;
        let mut dispatches = Vec::new();
        for record in records {
            if record.record_type != RecordType::C3ul {
                continue;
            }
            let dispatch = match record.parse_dispatch_record() {
                Ok(dispatch) => dispatch,
                Err(error) => {
                    tracing::debug!(offset = record.offset, size = record.size, %error, "skipping dispatch record");
                    continue;
                }
            };
            dispatches.push(DispatchCall {
                index: dispatches.len(),
                offset: record.offset,
                encoder_id: Some(dispatch.encoder_id),
                pipeline_addr: None,
                kernel_name: None,
                buffers: Vec::new(),
                grid_size: dispatch.grid_size,
                group_size: dispatch.group_size,
            });
        }
        Ok(dispatches)
    }

    pub fn command_buffer_regions(&self) -> Result<Vec<CommandBufferRegion>> {
        let capture = self.capture_data()?;
        let command_buffers = parse_command_buffers(&capture);
        let encoders = dedupe_encoders(parse_compute_encoders(&capture))?;
        let buffer_names = self.buffer_name_map()?;
        let pipeline_events = parse_pipeline_state_events(&capture, &buffer_names)?;
        let dispatches = self.dispatch_calls()?;
        Ok(build_command_buffer_regions(
            &capture,
            command_buffers,
            encoders,
            pipeline_events,
            dispatches,
            &self.pipeline_function_map()?,
        ))
    }

    pub fn analyze_kernels(&self) -> Result<BTreeMap<String, KernelStat>> {
        let pipeline_map = self.pipeline_function_map()?;
        let regions = self.command_buffer_regions()?;
        let mut stats = BTreeMap::new();

        for (addr, name) in &pipeline_map {
            stats.entry(name.clone()).or_insert_with(|| KernelStat {
                name: name.clone(),
                pipeline_addr: *addr,
                dispatch_count: 0,
                encoder_labels: BTreeMap::new(),
                buffers: BTreeMap::new(),
            });
        }

        for region in regions {
            let encoder_by_addr: BTreeMap<u64, &ComputeEncoder> = region
                .encoders
                .iter()
                .map(|encoder| (encoder.address, encoder))
                .collect();

            for dispatch in region.dispatches {
                let encoder = dispatch
                    .encoder_id
                    .and_then(|encoder_id| encoder_by_addr.get(&encoder_id).copied())
                    .or_else(|| {
                        region
                            .encoders
                            .iter()
                            .rev()
                            .find(|encoder| encoder.offset <= dispatch.offset)
                    });

                let name = dispatch
                    .kernel_name
                    .clone()
                    .or_else(|| {
                        encoder
                            .map(|encoder| encoder.label.clone())
                            .filter(|label| !label.is_empty())
                    })
                    .unwrap_or_else(|| "unknown".to_owned());

                let pipeline_addr = dispatch.pipeline_addr.unwrap_or_else(|| {
                    encoder
                        .and_then(|encoder| {
                            pipeline_map.iter().find_map(|(addr, candidate)| {
                                (candidate == &encoder.label).then_some(*addr)
                            })
                        })
                        .unwrap_or_default()
                });

                let stat = stats.entry(name.clone()).or_insert_with(|| KernelStat {
                    name: name.clone(),
                    pipeline_addr,
                    dispatch_count: 0,
                    encoder_labels: BTreeMap::new(),
                    buffers: BTreeMap::new(),
                });
                stat.dispatch_count += 1;
                if let Some(encoder) = encoder
                    && !encoder.label.is_empty()
                {
                    *stat
                        .encoder_labels
                        .entry(encoder.label.clone())
                        .or_default() += 1;
                }
                for buffer in &dispatch.buffers {
                    let key = buffer
                        .name
                        .clone()
                        .unwrap_or_else(|| format!("0x{:x}", buffer.address));
                    *stat.buffers.entry(key).or_default() += 1;
                }
            }
        }

        if stats
            .get("unknown")
            .is_some_and(|entry| entry.dispatch_count == 0)
        {
            stats.remove("unknown");
        }

        Ok(stats)
    }

    pub fn buffer_name_map(&self) -> Result<BTreeMap<u64, String>> {
        let capture = self.capture_data()?;
        let mut names = BTreeMap::new();
        collect_buffer_names_from_data(&capture, &mut names)?;
        for resource in &self.device_resources {
            let data = fs::read(&resource.path)?;
            collect_buffer_names_from_data(&data, &mut names)?;
        }
        Ok(names)
    }

    pub fn analyze_buffers(&self) -> Result<BTreeMap<String, BufferAccessStat>> {
        let regions = self.command_buffer_regions()?;
        let mut stats = BTreeMap::new();
        let mut encoder_sets: BTreeMap<String, BTreeSet<u64>> = BTreeMap::new();
        let mut command_buffer_sets: BTreeMap<String, BTreeSet<usize>> = BTreeMap::new();

        for region in regions {
            for dispatch in region.dispatches {
                let kernel_name = dispatch
                    .kernel_name
                    .clone()
                    .unwrap_or_else(|| "unknown".to_owned());
                for buffer in dispatch.buffers {
                    let name = buffer
                        .name
                        .clone()
                        .unwrap_or_else(|| format!("0x{:x}", buffer.address));
                    let entry = stats
                        .entry(name.clone())
                        .or_insert_with(|| BufferAccessStat {
                            name: name.clone(),
                            address: Some(buffer.address),
                            use_count: 0,
                            dispatch_count: 0,
                            encoder_count: 0,
                            command_buffer_count: 0,
                            first_dispatch_index: dispatch.index,
                            last_dispatch_index: dispatch.index,
                            kernels: BTreeMap::new(),
                        });
                    entry.use_count += 1;
                    entry.dispatch_count += 1;
                    entry.first_dispatch_index = entry.first_dispatch_index.min(dispatch.index);
                    entry.last_dispatch_index = entry.last_dispatch_index.max(dispatch.index);
                    *entry.kernels.entry(kernel_name.clone()).or_default() += 1;
                    command_buffer_sets
                        .entry(name.clone())
                        .or_default()
                        .insert(region.command_buffer.index);
                    if let Some(encoder_id) = dispatch.encoder_id {
                        encoder_sets.entry(name).or_default().insert(encoder_id);
                    }
                }
            }
        }

        for (name, entry) in &mut stats {
            entry.encoder_count = encoder_sets.get(name).map_or(0, BTreeSet::len);
            entry.command_buffer_count = command_buffer_sets.get(name).map_or(0, BTreeSet::len);
        }

        Ok(stats)
    }

    pub fn analyze_buffer_lifecycles(&self) -> Result<BTreeMap<String, BufferLifecycleStat>> {
        let regions = self.command_buffer_regions()?;
        Ok(analyze_buffer_lifecycles_from_regions(&regions))
    }
}

fn parse_metadata(bundle_path: &Path) -> Result<Metadata> {
    let metadata_path = bundle_path.join("metadata");
    if !metadata_path.exists() {
        return Err(Error::MissingFile(metadata_path));
    }
    let plist = Value::from_file(&metadata_path)?;
    let Some(dict) = plist.as_dictionary() else {
        return Err(Error::InvalidTrace("metadata plist was not a dictionary"));
    };

    let mut metadata = Metadata {
        uuid: dict.get("(uuid)").and_then(as_string),
        capture_version: dict
            .get("DYCaptureSession.capture_version")
            .and_then(as_integer),
        graphics_api: dict
            .get("DYCaptureSession.graphics_api")
            .and_then(as_integer),
        device_id: dict.get("DYCaptureSession.deviceId").and_then(as_integer),
        native_pointer_size: dict
            .get("DYCaptureSession.nativePointerSize")
            .and_then(as_integer),
        captured_frames_count: dict
            .get("DYCaptureEngine.captured_frames_count")
            .and_then(as_integer),
        boundary_less: dict
            .get("DYCaptureSession.boundaryLess")
            .and_then(Value::as_boolean),
        library_link_versions: BTreeMap::new(),
        unused_buffer_count: dict
            .get("DYCaptureSession.unusedBufferCount")
            .and_then(as_integer),
        unused_texture_count: dict
            .get("DYCaptureSession.unusedTextureCount")
            .and_then(as_integer),
        unused_function_count: dict
            .get("DYCaptureSession.unusedFunctionCount")
            .and_then(as_integer),
    };

    if let Some(libraries) = dict
        .get("DYCaptureSession.library_link_time_versions")
        .and_then(Value::as_dictionary)
    {
        for (name, value) in libraries {
            if let Some(version) = as_integer(value) {
                metadata.library_link_versions.insert(name.clone(), version);
            }
        }
    }

    Ok(metadata)
}

fn load_capture_path(bundle_path: &Path) -> Result<PathBuf> {
    if let Some(path) = local_capture_path(bundle_path)? {
        return Ok(path);
    }

    if let Some(path) = sibling_capture_path_for_perfdata(bundle_path)? {
        return Ok(path);
    }

    Err(Error::MissingFile(bundle_path.join("capture")))
}

fn local_capture_path(bundle_path: &Path) -> Result<Option<PathBuf>> {
    for candidate in ["capture", "unsorted-capture"] {
        let path = bundle_path.join(candidate);
        if path.exists() {
            let bytes = fs::read(&path)?;
            if bytes.get(..4) != Some(MAGIC_MTSP.as_slice()) {
                return Err(Error::InvalidTrace("capture file did not start with MTSP"));
            }
            return Ok(Some(path));
        }
    }
    Ok(None)
}

fn sibling_capture_path_for_perfdata(bundle_path: &Path) -> Result<Option<PathBuf>> {
    let Some(parent) = bundle_path.parent() else {
        return Ok(None);
    };
    let Ok(entries) = fs::read_dir(bundle_path) else {
        return Ok(None);
    };

    for entry in entries {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        let Some(trace_name) = name.strip_suffix(".gpuprofiler_raw") else {
            continue;
        };
        let sibling = parent.join(trace_name);
        if sibling == bundle_path {
            continue;
        }
        if let Some(path) = local_capture_path(&sibling)? {
            return Ok(Some(path));
        }
    }

    Ok(None)
}

fn load_device_resources(bundle_path: &Path) -> Result<Vec<DeviceResource>> {
    let mut resources = Vec::new();
    for entry in fs::read_dir(bundle_path)? {
        let entry = entry?;
        if !entry.file_type()?.is_file() {
            continue;
        }
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        if !name.starts_with("device-resources-") {
            continue;
        }
        let path = entry.path();
        resources.push(DeviceResource {
            len: fs::metadata(&path)?.len(),
            path,
        });
    }
    resources.sort_by(|left, right| left.path.cmp(&right.path));
    Ok(resources)
}

fn as_string(value: &Value) -> Option<String> {
    value.as_string().map(ToOwned::to_owned)
}

fn as_integer(value: &Value) -> Option<i64> {
    match value {
        Value::Integer(value) => value
            .as_signed()
            .or_else(|| value.as_unsigned().map(|v| v as i64)),
        Value::Real(value) => Some(*value as i64),
        _ => None,
    }
}

fn collect_labels_from_data(
    data: &[u8],
    labels: &mut BTreeMap<u64, String>,
    blob_refs: &mut BTreeMap<u64, String>,
) -> Result<()> {
    let records = MTSPRecord::parse_stream(data)?;
    for record in records {
        let (Some(receiver), Some(label)) = (record.address, record.label.clone()) else {
            continue;
        };
        // A factory call names the object it returns (a function made by its
        // library); a `setLabel:` names its receiver.
        let address = record.returned_address().unwrap_or(receiver);
        match record.record_type {
            RecordType::CS => {
                labels.insert(address, label);
            }
            RecordType::CUt => {
                blob_refs.insert(address, label);
            }
            _ => {}
        }
    }
    Ok(())
}

/// The name of a specialized function from its descriptor blob: a 16-byte
/// header, then the function name and the specialized name as NUL-padded C
/// strings, then the constant values. The specialized name, which spells out
/// the constants, wins when present.
fn function_descriptor_name(blob: &[u8]) -> Option<String> {
    let c_string = |start: usize| -> Option<(String, usize)> {
        let rest = blob.get(start..)?;
        let end = rest.iter().position(|&byte| byte == 0)?;
        let text = std::str::from_utf8(&rest[..end]).ok()?;
        let is_symbol = !text.is_empty()
            && text
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_');
        is_symbol.then(|| (text.to_owned(), start + end))
    };
    let (name, end) = c_string(16)?;
    let specialized_start = blob[end..].iter().position(|&byte| byte != 0)? + end;
    match c_string(specialized_start) {
        Some((specialized, _)) if specialized.starts_with(&name) => Some(specialized),
        _ => Some(name),
    }
}

fn collect_pipeline_mappings_from_data(
    data: &[u8],
    labels: &BTreeMap<u64, String>,
    result: &mut PipelineFunctionMap,
) -> Result<()> {
    let records = MTSPRecord::parse_stream(data)?;
    for record in records {
        if record.record_type != RecordType::Ctt {
            continue;
        }
        if let Ok(ctt) = record.parse_ctt_record() {
            let label = labels.get(&ctt.function_addr);
            tracing::trace!(
                pipeline = format_args!("{:#x}", ctt.pipeline_addr),
                function = format_args!("{:#x}", ctt.function_addr),
                label = label.map(String::as_str).unwrap_or("-"),
                "pipeline"
            );
            if let Some(label) = label {
                result.insert(ctt.pipeline_addr, label.clone());
            }
        }
    }
    Ok(())
}

fn collect_buffer_names_from_data(data: &[u8], names: &mut BTreeMap<u64, String>) -> Result<()> {
    let records = MTSPRecord::parse_stream(data)?;
    for record in records {
        if record.record_type != RecordType::CtU {
            continue;
        }
        // One malformed record (seen: short CtU records) must not cost
        // every section that names buffers.
        let ctu = match record.parse_ctu_record() {
            Ok(ctu) => ctu,
            Err(error) => {
                tracing::debug!(offset = record.offset, size = record.size, %error, "skipping CtU record");
                continue;
            }
        };
        names.insert(ctu.address, ctu.name);
    }
    Ok(())
}

fn parse_command_buffers(data: &[u8]) -> Vec<CommandBuffer> {
    // CUUU is a labeled-resource sub-record (same nested framing as CS) carried
    // inside a top-level bulk record's payload. The previous byte-scan read a
    // u64 at marker+4 and called it the timestamp, but on this trace those
    // bytes are actually the 4-byte alignment pad after the tag plus the low
    // half of the following u64 address — i.e., the value was always bogus.
    // Real per-CB timestamps come from APSTimelineData (see
    // `command_buffer_timings_from_timeline` in `timing.rs`); we leave
    // `timestamp` as 0 here and let the profiler-backed paths fill it in.
    let Ok(records) = MTSPRecord::parse_stream(data) else {
        return Vec::new();
    };
    let mut command_buffers = Vec::new();
    let mut seen_offsets: BTreeSet<usize> = BTreeSet::new();
    let mut push = |cuuu: &MTSPRecord, offset: usize, dst: &mut Vec<CommandBuffer>| {
        if !seen_offsets.insert(offset) {
            return;
        }
        dst.push(CommandBuffer {
            index: dst.len(),
            timestamp: 0,
            offset,
            artifact_hashes: parse_cuuu_hashes(&cuuu.data),
        });
    };
    for record in records {
        if record.record_type == RecordType::CUUU {
            push(&record, record.offset, &mut command_buffers);
        }
        for nested in MTSPRecord::parse_subrecords(&record.data) {
            if nested.record_type == RecordType::CUUU {
                push(&nested, record.offset + nested.offset, &mut command_buffers);
            }
        }
    }
    for top_level_subrecord in MTSPRecord::parse_subrecords(data) {
        if top_level_subrecord.record_type == RecordType::CUUU {
            push(
                &top_level_subrecord,
                top_level_subrecord.offset,
                &mut command_buffers,
            );
        }
    }
    command_buffers
}

/// Extract artifact hash strings from a CUUU sub-record's payload.
///
/// The sub-record layout is `[u32 size][4 magic][24 zeros][u32 count][CUUU][4
/// zeros][u64 address][c-strings...]`. The c-strings are 16-character
/// uppercase hex hashes that name per-CB artifact subdirectories of the
/// trace bundle.
fn parse_cuuu_hashes(data: &[u8]) -> Vec<String> {
    // First c-string starts at offset 52 (size + magic + 24 zeros + count + 4
    // tag + 4 alignment + 8 address = 52).
    let mut hashes = Vec::new();
    let mut pos = 52;
    while pos < data.len() {
        let end = data[pos..]
            .iter()
            .position(|&byte| byte == 0)
            .map(|delta| pos + delta)
            .unwrap_or(data.len());
        if end == pos {
            pos += 1;
            continue;
        }
        let candidate = &data[pos..end];
        let is_hex = !candidate.is_empty()
            && candidate
                .iter()
                .all(|&byte| byte.is_ascii_hexdigit() && !byte.is_ascii_lowercase());
        if is_hex && (12..=32).contains(&candidate.len()) {
            hashes.push(String::from_utf8_lossy(candidate).into_owned());
        }
        pos = end + 1;
    }
    hashes
}

fn parse_compute_encoders(data: &[u8]) -> Vec<ComputeEncoder> {
    // CS is the labeled-resource record family — compute encoders, command
    // buffers, and MTLBuffers all share the `CS\0\0` tag. The capture format
    // varies: the offline-driver bee traces wrap CS sub-records inside bulk
    // CtU/Unknown records, while programmatic `MTLCaptureManager` traces
    // surface the sub-records at top level with the same nested framing. We
    // walk both: top-level MTSP records first, then the nested sub-record
    // streams inside each record's payload, then a top-level sub-record scan
    // to pick up captures that don't use the bulk-record wrapper.
    //
    // To discriminate compute encoders from MTLBuffers, look for a CtU
    // sibling: every MTLBuffer in this format is announced by a CtU record
    // (with a name like `MTLBuffer-1681-0`) at the same address as its CS
    // label. Compute encoders and command buffers don't get that
    // announcement. So `CS without matching CtU == compute infrastructure`.
    let Ok(records) = MTSPRecord::parse_stream(data) else {
        return Vec::new();
    };

    let mut buffer_addresses: BTreeSet<u64> = BTreeSet::new();
    let visit_ctu = |record: &MTSPRecord, dst: &mut BTreeSet<u64>| {
        if record.record_type == RecordType::CtU
            && let Ok(ctu) = record.parse_ctu_record()
        {
            dst.insert(ctu.address);
        }
    };
    for record in &records {
        visit_ctu(record, &mut buffer_addresses);
        for nested in MTSPRecord::parse_subrecords(&record.data) {
            visit_ctu(&nested, &mut buffer_addresses);
        }
    }
    for top_level_subrecord in MTSPRecord::parse_subrecords(data) {
        visit_ctu(&top_level_subrecord, &mut buffer_addresses);
    }

    let mut encoders = Vec::new();
    let mut seen_offsets: BTreeSet<usize> = BTreeSet::new();
    let mut push_if_encoder =
        |address: u64, label: Option<String>, offset: usize, encoders: &mut Vec<ComputeEncoder>| {
            let Some(label) = label else { return };
            if label.is_empty() || buffer_addresses.contains(&address) {
                return;
            }
            if !seen_offsets.insert(offset) {
                return;
            }
            encoders.push(ComputeEncoder {
                index: encoders.len(),
                address,
                label,
                offset,
            });
        };

    for record in records {
        if record.record_type == RecordType::CS
            && let Some(address) = record.address
        {
            push_if_encoder(address, record.label.clone(), record.offset, &mut encoders);
        }
        for nested in MTSPRecord::parse_subrecords(&record.data) {
            if nested.record_type == RecordType::CS
                && let Some(address) = nested.address
            {
                push_if_encoder(
                    address,
                    nested.label,
                    record.offset + nested.offset,
                    &mut encoders,
                );
            }
        }
    }
    for top_level_subrecord in MTSPRecord::parse_subrecords(data) {
        if top_level_subrecord.record_type == RecordType::CS
            && let Some(address) = top_level_subrecord.address
        {
            push_if_encoder(
                address,
                top_level_subrecord.label,
                top_level_subrecord.offset,
                &mut encoders,
            );
        }
    }
    encoders
}

fn parse_pipeline_state_events(
    data: &[u8],
    buffer_names: &BTreeMap<u64, String>,
) -> Result<Vec<PipelineStateEvent>> {
    let records = MTSPRecord::parse_stream(data)?;
    let mut events = Vec::new();
    for record in records {
        if record.record_type != RecordType::Ct {
            continue;
        }
        let ct = match record.parse_ct_record() {
            Ok(ct) => ct,
            Err(error) => {
                tracing::debug!(offset = record.offset, size = record.size, %error, "skipping Ct record");
                continue;
            }
        };
        events.push(PipelineStateEvent {
            offset: record.offset,
            encoder_addr: ct.function_addr,
            pipeline_addr: ct.pipeline_addr,
            function_addr: ct.function_addr,
            buffers: ct
                .resource_bindings
                .into_iter()
                .map(|binding| BoundBuffer {
                    address: binding.address,
                    name: buffer_names.get(&binding.address).cloned(),
                    index: binding.index,
                    usage: binding.usage,
                })
                .collect(),
        });
    }
    Ok(events)
}

fn dedupe_encoders(mut encoders: Vec<ComputeEncoder>) -> Result<Vec<ComputeEncoder>> {
    encoders.sort_by_key(|encoder| (encoder.offset, encoder.address));
    encoders.dedup_by(|left, right| left.offset == right.offset && left.address == right.address);
    for (index, encoder) in encoders.iter_mut().enumerate() {
        encoder.index = index;
    }
    Ok(encoders)
}

fn build_command_buffer_regions(
    capture: &[u8],
    command_buffers: Vec<CommandBuffer>,
    encoders: Vec<ComputeEncoder>,
    pipeline_events: Vec<PipelineStateEvent>,
    dispatches: Vec<DispatchCall>,
    pipeline_map: &PipelineFunctionMap,
) -> Vec<CommandBufferRegion> {
    if command_buffers.is_empty() {
        return vec![CommandBufferRegion {
            command_buffer: CommandBuffer {
                index: 0,
                timestamp: 0,
                offset: 0,
                artifact_hashes: Vec::new(),
            },
            end_offset: capture.len(),
            encoders,
            pipeline_events: pipeline_events.clone(),
            dispatches: attribute_dispatches(dispatches, &pipeline_events, pipeline_map),
        }];
    }

    let mut regions = Vec::new();
    for (index, command_buffer) in command_buffers.iter().cloned().enumerate() {
        let end_offset = command_buffers
            .get(index + 1)
            .map(|next| next.offset)
            .unwrap_or(capture.len());
        let region_encoders = encoders
            .iter()
            .filter(|encoder| {
                encoder.offset >= command_buffer.offset && encoder.offset < end_offset
            })
            .cloned()
            .collect();
        let region_pipeline_events: Vec<_> = pipeline_events
            .iter()
            .filter(|event| event.offset >= command_buffer.offset && event.offset < end_offset)
            .cloned()
            .collect();
        let region_dispatches = dispatches
            .iter()
            .filter(|dispatch| {
                dispatch.offset >= command_buffer.offset && dispatch.offset < end_offset
            })
            .cloned()
            .collect::<Vec<_>>();
        regions.push(CommandBufferRegion {
            command_buffer,
            end_offset,
            encoders: region_encoders,
            pipeline_events: region_pipeline_events.clone(),
            dispatches: attribute_dispatches(
                region_dispatches,
                &region_pipeline_events,
                pipeline_map,
            ),
        });
    }
    regions
}

fn attribute_dispatches(
    mut dispatches: Vec<DispatchCall>,
    pipeline_events: &[PipelineStateEvent],
    pipeline_map: &PipelineFunctionMap,
) -> Vec<DispatchCall> {
    for dispatch in &mut dispatches {
        let event = pipeline_events
            .iter()
            .rev()
            .find(|event| {
                event.offset <= dispatch.offset && Some(event.encoder_addr) == dispatch.encoder_id
            })
            .or_else(|| {
                pipeline_events
                    .iter()
                    .rev()
                    .find(|event| event.offset <= dispatch.offset)
            });
        if let Some(event) = event {
            dispatch.pipeline_addr = Some(event.pipeline_addr);
            dispatch.kernel_name = pipeline_map.get(&event.pipeline_addr).cloned();
            dispatch.buffers = event.buffers.clone();
            if dispatch.encoder_id.is_none() {
                dispatch.encoder_id = Some(event.encoder_addr);
            }
        }
    }
    dispatches
}

fn analyze_buffer_lifecycles_from_regions(
    regions: &[CommandBufferRegion],
) -> BTreeMap<String, BufferLifecycleStat> {
    let mut stats = BTreeMap::new();
    let mut encoder_sets: BTreeMap<String, BTreeSet<u64>> = BTreeMap::new();

    for region in regions {
        for dispatch in &region.dispatches {
            let kernel_name = dispatch
                .kernel_name
                .clone()
                .unwrap_or_else(|| "unknown".to_owned());
            for buffer in &dispatch.buffers {
                let name = buffer
                    .name
                    .clone()
                    .unwrap_or_else(|| format!("0x{:x}", buffer.address));
                let entry = stats
                    .entry(name.clone())
                    .or_insert_with(|| BufferLifecycleStat {
                        name: name.clone(),
                        address: Some(buffer.address),
                        first_command_buffer_index: region.command_buffer.index,
                        last_command_buffer_index: region.command_buffer.index,
                        first_dispatch_index: dispatch.index,
                        last_dispatch_index: dispatch.index,
                        command_buffer_span: 1,
                        dispatch_span: 1,
                        use_count: 0,
                        encoder_count: 0,
                        kernels: BTreeMap::new(),
                    });
                entry.first_command_buffer_index = entry
                    .first_command_buffer_index
                    .min(region.command_buffer.index);
                entry.last_command_buffer_index = entry
                    .last_command_buffer_index
                    .max(region.command_buffer.index);
                entry.first_dispatch_index = entry.first_dispatch_index.min(dispatch.index);
                entry.last_dispatch_index = entry.last_dispatch_index.max(dispatch.index);
                entry.command_buffer_span =
                    entry.last_command_buffer_index - entry.first_command_buffer_index + 1;
                entry.dispatch_span = entry.last_dispatch_index - entry.first_dispatch_index + 1;
                entry.use_count += 1;
                *entry.kernels.entry(kernel_name.clone()).or_default() += 1;
                if let Some(encoder_id) = dispatch.encoder_id {
                    encoder_sets.entry(name).or_default().insert(encoder_id);
                }
            }
        }
    }

    for (name, entry) in &mut stats {
        entry.encoder_count = encoder_sets.get(name).map_or(0, BTreeSet::len);
    }

    stats
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_metadata(path: &Path) {
        fs::write(
            path.join("metadata"),
            r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "https://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict><key>(uuid)</key><string>test</string></dict></plist>
"#,
        )
        .unwrap();
    }

    #[test]
    fn opens_perfdata_export_using_sibling_raw_capture() {
        let dir = tempfile::tempdir().unwrap();
        let raw = dir.path().join("raw.gputrace");
        let export = dir.path().join("raw-perfdata.gputrace");
        fs::create_dir_all(&raw).unwrap();
        fs::create_dir_all(&export).unwrap();
        write_metadata(&raw);
        write_metadata(&export);
        fs::write(raw.join("capture"), [b"MTSP".as_slice(), &[0; 12]].concat()).unwrap();
        fs::write(raw.join("device-resources-0"), b"resource").unwrap();
        fs::create_dir_all(export.join("raw.gputrace.gpuprofiler_raw")).unwrap();
        fs::write(export.join("store0"), b"packed").unwrap();
        fs::write(export.join("index"), b"index").unwrap();

        let bundle = TraceBundle::open(&export).unwrap();

        assert_eq!(bundle.path, export);
        assert_eq!(bundle.capture_path, raw.join("capture"));
        assert_eq!(bundle.device_resources.len(), 1);
        assert_eq!(
            bundle.device_resources[0].path,
            raw.join("device-resources-0")
        );
    }

    #[test]
    fn parses_command_buffer_markers() {
        // CUUU sub-records are nested inside a bulk top-level record. Build a
        // 0x140-byte top-level record whose payload contains two nested CUUU
        // sub-records (each 0x68 bytes) at relative offsets 0x40 and 0xb0.
        let bulk_size = 0x140usize;
        let mut bulk = vec![0u8; bulk_size];
        bulk[0..4].copy_from_slice(&(bulk_size as u32).to_le_bytes());
        bulk[8..12].copy_from_slice(b"Ctt\0");

        for (sub_off, seq) in [(0x40usize, 0x07u8), (0xb0usize, 0x0bu8)] {
            let sub_size = 0x68usize;
            bulk[sub_off..sub_off + 4].copy_from_slice(&(sub_size as u32).to_le_bytes());
            bulk[sub_off + 4..sub_off + 8].copy_from_slice(&[seq, 0xd8, 0xff, 0xff]);
            // bytes [sub_off + 8 .. sub_off + 32] stay zero
            bulk[sub_off + 32..sub_off + 36].copy_from_slice(&4u32.to_le_bytes());
            bulk[sub_off + 36..sub_off + 40].copy_from_slice(b"CUUU");
        }

        let buffers = parse_command_buffers(&bulk);
        assert_eq!(buffers.len(), 2);
        assert_eq!(buffers[0].index, 0);
        assert_eq!(buffers[1].index, 1);
        // timestamps come from APSTimelineData, not CUUU bytes.
        assert_eq!(buffers[0].timestamp, 0);
        assert_eq!(buffers[1].timestamp, 0);
        assert!(buffers[0].offset < buffers[1].offset);
    }

    #[test]
    fn parses_compute_encoders_from_cs_records() {
        // Top-level CS record: 32-byte framed MTSP record with the CS tag in
        // the first 128 bytes.
        let mut record = vec![0u8; 32];
        record[0..4].copy_from_slice(&(32u32).to_le_bytes());
        record[8..12].copy_from_slice(b"CS\0\0");
        record[12..20].copy_from_slice(&0x1234u64.to_le_bytes());
        record[20..27].copy_from_slice(b"Kernel\0");

        let encoders = parse_compute_encoders(&record);
        assert_eq!(encoders.len(), 1);
        assert_eq!(encoders[0].address, 0x1234);
        assert_eq!(encoders[0].label, "Kernel");
    }

    #[test]
    fn parses_compute_encoders_from_nested_cs_subrecords() {
        // Bulk top-level record (size 0x100) whose payload contains one nested
        // CS sub-record at offset 0x40 of the bulk record.
        let bulk_size = 0x100usize;
        let mut bulk = vec![0u8; bulk_size];
        bulk[0..4].copy_from_slice(&(bulk_size as u32).to_le_bytes());
        // Tag the bulk record as Ctt so it doesn't get classified as CS itself.
        bulk[8..12].copy_from_slice(b"Ctt\0");

        // Nested sub-record at relative offset 0x40 inside the bulk record:
        //   size=0xb0, magic=`?? c0 ff ff`, 24 zero pad, count=0xc, tag=CS\0\0,
        //   address u64, c-string label.
        let sub_off = 0x40usize;
        let sub_size = 0xb0usize;
        bulk[sub_off..sub_off + 4].copy_from_slice(&(sub_size as u32).to_le_bytes());
        bulk[sub_off + 4..sub_off + 8].copy_from_slice(&[0x05, 0xc0, 0xff, 0xff]);
        // Bytes [sub_off + 8 .. sub_off + 32] stay zero (already initialised).
        bulk[sub_off + 32..sub_off + 36].copy_from_slice(&12u32.to_le_bytes());
        bulk[sub_off + 36..sub_off + 40].copy_from_slice(b"CS\0\0");
        bulk[sub_off + 40..sub_off + 48].copy_from_slice(&0xdeadbeefu64.to_le_bytes());
        let label = b"NestedEncoder";
        bulk[sub_off + 48..sub_off + 48 + label.len()].copy_from_slice(label);

        let encoders = parse_compute_encoders(&bulk);
        assert_eq!(encoders.len(), 1);
        assert_eq!(encoders[0].address, 0xdeadbeef);
        assert_eq!(encoders[0].label, "NestedEncoder");
    }

    #[test]
    fn skips_short_ctt_records_when_collecting_pipeline_mappings() {
        let mut data = Vec::new();
        let mut short_ctt = vec![0u8; 24];
        short_ctt[0..4].copy_from_slice(&(24u32).to_le_bytes());
        short_ctt[8..12].copy_from_slice(b"Ctt\0");
        data.extend_from_slice(&short_ctt);

        let mut full_ctt = vec![0u8; 64];
        full_ctt[0..4].copy_from_slice(&(64u32).to_le_bytes());
        full_ctt[8..12].copy_from_slice(b"Ctt\0");
        full_ctt[20..28].copy_from_slice(&0x7000u64.to_le_bytes());
        full_ctt[40..48].copy_from_slice(&0x9000u64.to_le_bytes());
        data.extend_from_slice(&full_ctt);

        let mut labels = BTreeMap::new();
        labels.insert(0x7000, "kernel_main".to_owned());
        let mut mappings = BTreeMap::new();

        collect_pipeline_mappings_from_data(&data, &labels, &mut mappings).unwrap();

        assert_eq!(
            mappings.get(&0x9000).map(String::as_str),
            Some("kernel_main")
        );
    }

    #[test]
    fn builds_command_buffer_regions() {
        let command_buffers = vec![
            CommandBuffer {
                index: 0,
                timestamp: 1,
                offset: 10,
                artifact_hashes: Vec::new(),
            },
            CommandBuffer {
                index: 1,
                timestamp: 2,
                offset: 50,
                artifact_hashes: Vec::new(),
            },
        ];
        let encoders = vec![
            ComputeEncoder {
                index: 0,
                address: 1,
                label: "a".into(),
                offset: 20,
            },
            ComputeEncoder {
                index: 1,
                address: 2,
                label: "b".into(),
                offset: 60,
            },
        ];
        let dispatches = vec![
            DispatchCall {
                index: 0,
                offset: 30,
                encoder_id: Some(1),
                pipeline_addr: None,
                kernel_name: None,
                buffers: Vec::new(),
                grid_size: [1, 1, 1],
                group_size: [1, 1, 1],
            },
            DispatchCall {
                index: 1,
                offset: 70,
                encoder_id: Some(2),
                pipeline_addr: None,
                kernel_name: None,
                buffers: Vec::new(),
                grid_size: [1, 1, 1],
                group_size: [1, 1, 1],
            },
        ];
        let pipeline_events = vec![
            PipelineStateEvent {
                offset: 25,
                encoder_addr: 1,
                pipeline_addr: 10,
                function_addr: 1,
                buffers: vec![BoundBuffer {
                    address: 0xaa,
                    name: Some("ba".into()),
                    index: 0,
                    usage: MTLResourceUsage::READ | MTLResourceUsage::WRITE,
                }],
            },
            PipelineStateEvent {
                offset: 65,
                encoder_addr: 2,
                pipeline_addr: 20,
                function_addr: 2,
                buffers: vec![BoundBuffer {
                    address: 0xbb,
                    name: Some("bb".into()),
                    index: 1,
                    usage: MTLResourceUsage::READ | MTLResourceUsage::WRITE,
                }],
            },
        ];
        let mut pipeline_map = BTreeMap::new();
        pipeline_map.insert(10, "ka".into());
        pipeline_map.insert(20, "kb".into());

        let regions = build_command_buffer_regions(
            &[0; 100],
            command_buffers,
            encoders,
            pipeline_events,
            dispatches,
            &pipeline_map,
        );
        assert_eq!(regions.len(), 2);
        assert_eq!(regions[0].encoders.len(), 1);
        assert_eq!(regions[0].pipeline_events.len(), 1);
        assert_eq!(regions[0].dispatches.len(), 1);
        assert_eq!(regions[0].dispatches[0].kernel_name.as_deref(), Some("ka"));
        assert_eq!(
            regions[0].dispatches[0].buffers[0].name.as_deref(),
            Some("ba")
        );
        assert_eq!(regions[1].encoders.len(), 1);
        assert_eq!(regions[1].pipeline_events.len(), 1);
        assert_eq!(regions[1].dispatches.len(), 1);
        assert_eq!(regions[1].dispatches[0].kernel_name.as_deref(), Some("kb"));
        assert_eq!(
            regions[1].dispatches[0].buffers[0].name.as_deref(),
            Some("bb")
        );
    }

    #[test]
    fn analyzes_buffer_lifecycles_from_regions() {
        let regions = vec![
            CommandBufferRegion {
                command_buffer: CommandBuffer {
                    index: 0,
                    timestamp: 1,
                    offset: 0,
                    artifact_hashes: Vec::new(),
                },
                end_offset: 10,
                encoders: vec![],
                pipeline_events: vec![],
                dispatches: vec![
                    DispatchCall {
                        index: 0,
                        offset: 1,
                        encoder_id: Some(1),
                        pipeline_addr: Some(10),
                        kernel_name: Some("ka".into()),
                        buffers: vec![BoundBuffer {
                            address: 0xaa,
                            name: Some("ba".into()),
                            index: 0,
                            usage: MTLResourceUsage::READ | MTLResourceUsage::WRITE,
                        }],
                        grid_size: [1, 1, 1],
                        group_size: [1, 1, 1],
                    },
                    DispatchCall {
                        index: 1,
                        offset: 2,
                        encoder_id: Some(1),
                        pipeline_addr: Some(10),
                        kernel_name: Some("kb".into()),
                        buffers: vec![BoundBuffer {
                            address: 0xbb,
                            name: Some("bb".into()),
                            index: 1,
                            usage: MTLResourceUsage::READ | MTLResourceUsage::WRITE,
                        }],
                        grid_size: [1, 1, 1],
                        group_size: [1, 1, 1],
                    },
                ],
            },
            CommandBufferRegion {
                command_buffer: CommandBuffer {
                    index: 2,
                    timestamp: 2,
                    offset: 10,
                    artifact_hashes: Vec::new(),
                },
                end_offset: 20,
                encoders: vec![],
                pipeline_events: vec![],
                dispatches: vec![DispatchCall {
                    index: 4,
                    offset: 12,
                    encoder_id: Some(2),
                    pipeline_addr: Some(20),
                    kernel_name: Some("ka".into()),
                    buffers: vec![BoundBuffer {
                        address: 0xaa,
                        name: Some("ba".into()),
                        index: 0,
                        usage: MTLResourceUsage::READ | MTLResourceUsage::WRITE,
                    }],
                    grid_size: [1, 1, 1],
                    group_size: [1, 1, 1],
                }],
            },
        ];
        let stats = analyze_buffer_lifecycles_from_regions(&regions);

        let ba = stats.get("ba").unwrap();
        assert_eq!(ba.first_command_buffer_index, 0);
        assert_eq!(ba.last_command_buffer_index, 2);
        assert_eq!(ba.command_buffer_span, 3);
        assert_eq!(ba.first_dispatch_index, 0);
        assert_eq!(ba.last_dispatch_index, 4);
        assert_eq!(ba.dispatch_span, 5);
        assert_eq!(ba.use_count, 2);
        assert_eq!(ba.kernels.get("ka"), Some(&2));

        let bb = stats.get("bb").unwrap();
        assert_eq!(bb.command_buffer_span, 1);
        assert_eq!(bb.dispatch_span, 1);
        assert_eq!(bb.use_count, 1);
    }
}
