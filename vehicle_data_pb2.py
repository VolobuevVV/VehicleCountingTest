# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: vehicle_data.proto
# Protobuf Python Version: 5.27.2
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    5,
    27,
    2,
    '',
    'vehicle_data.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x12vehicle_data.proto\x12\x0cvehicle_data\"1\n\tTimeRange\x12\x12\n\nstart_time\x18\x01 \x01(\x05\x12\x10\n\x08\x65nd_time\x18\x02 \x01(\x05\"@\n\x0bVehicleData\x12\x17\n\x0f\x63\x61r_number_left\x18\x01 \x01(\x05\x12\x18\n\x10\x63\x61r_number_right\x18\x02 \x01(\x05\x32V\n\x0eServiceVehicle\x12\x44\n\x0eGetVehicleData\x12\x17.vehicle_data.TimeRange\x1a\x19.vehicle_data.VehicleDatab\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'vehicle_data_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_TIMERANGE']._serialized_start=36
  _globals['_TIMERANGE']._serialized_end=85
  _globals['_VEHICLEDATA']._serialized_start=87
  _globals['_VEHICLEDATA']._serialized_end=151
  _globals['_SERVICEVEHICLE']._serialized_start=153
  _globals['_SERVICEVEHICLE']._serialized_end=239
# @@protoc_insertion_point(module_scope)
