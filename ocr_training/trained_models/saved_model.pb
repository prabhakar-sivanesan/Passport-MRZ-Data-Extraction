??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718??

{
dense_50/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?* 
shared_namedense_50/kernel
t
#dense_50/kernel/Read/ReadVariableOpReadVariableOpdense_50/kernel*
_output_shapes
:	?*
dtype0
s
dense_50/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_50/bias
l
!dense_50/bias/Read/ReadVariableOpReadVariableOpdense_50/bias*
_output_shapes	
:?*
dtype0
|
dense_51/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_51/kernel
u
#dense_51/kernel/Read/ReadVariableOpReadVariableOpdense_51/kernel* 
_output_shapes
:
??*
dtype0
s
dense_51/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_51/bias
l
!dense_51/bias/Read/ReadVariableOpReadVariableOpdense_51/bias*
_output_shapes	
:?*
dtype0
|
dense_52/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_52/kernel
u
#dense_52/kernel/Read/ReadVariableOpReadVariableOpdense_52/kernel* 
_output_shapes
:
??*
dtype0
s
dense_52/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_52/bias
l
!dense_52/bias/Read/ReadVariableOpReadVariableOpdense_52/bias*
_output_shapes	
:?*
dtype0
{
dense_53/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@* 
shared_namedense_53/kernel
t
#dense_53/kernel/Read/ReadVariableOpReadVariableOpdense_53/kernel*
_output_shapes
:	?@*
dtype0
r
dense_53/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_53/bias
k
!dense_53/bias/Read/ReadVariableOpReadVariableOpdense_53/bias*
_output_shapes
:@*
dtype0
|
dense_54/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_54/kernel
u
#dense_54/kernel/Read/ReadVariableOpReadVariableOpdense_54/kernel* 
_output_shapes
:
??*
dtype0
r
dense_54/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_54/bias
k
!dense_54/bias/Read/ReadVariableOpReadVariableOpdense_54/bias*
_output_shapes
:*
dtype0
l
RMSprop/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nameRMSprop/iter
e
 RMSprop/iter/Read/ReadVariableOpReadVariableOpRMSprop/iter*
_output_shapes
: *
dtype0	
n
RMSprop/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameRMSprop/decay
g
!RMSprop/decay/Read/ReadVariableOpReadVariableOpRMSprop/decay*
_output_shapes
: *
dtype0
~
RMSprop/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameRMSprop/learning_rate
w
)RMSprop/learning_rate/Read/ReadVariableOpReadVariableOpRMSprop/learning_rate*
_output_shapes
: *
dtype0
t
RMSprop/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameRMSprop/momentum
m
$RMSprop/momentum/Read/ReadVariableOpReadVariableOpRMSprop/momentum*
_output_shapes
: *
dtype0
j
RMSprop/rhoVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameRMSprop/rho
c
RMSprop/rho/Read/ReadVariableOpReadVariableOpRMSprop/rho*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
RMSprop/dense_50/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*,
shared_nameRMSprop/dense_50/kernel/rms
?
/RMSprop/dense_50/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_50/kernel/rms*
_output_shapes
:	?*
dtype0
?
RMSprop/dense_50/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:?**
shared_nameRMSprop/dense_50/bias/rms
?
-RMSprop/dense_50/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_50/bias/rms*
_output_shapes	
:?*
dtype0
?
RMSprop/dense_51/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*,
shared_nameRMSprop/dense_51/kernel/rms
?
/RMSprop/dense_51/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_51/kernel/rms* 
_output_shapes
:
??*
dtype0
?
RMSprop/dense_51/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:?**
shared_nameRMSprop/dense_51/bias/rms
?
-RMSprop/dense_51/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_51/bias/rms*
_output_shapes	
:?*
dtype0
?
RMSprop/dense_52/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*,
shared_nameRMSprop/dense_52/kernel/rms
?
/RMSprop/dense_52/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_52/kernel/rms* 
_output_shapes
:
??*
dtype0
?
RMSprop/dense_52/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:?**
shared_nameRMSprop/dense_52/bias/rms
?
-RMSprop/dense_52/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_52/bias/rms*
_output_shapes	
:?*
dtype0
?
RMSprop/dense_53/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*,
shared_nameRMSprop/dense_53/kernel/rms
?
/RMSprop/dense_53/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_53/kernel/rms*
_output_shapes
:	?@*
dtype0
?
RMSprop/dense_53/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_nameRMSprop/dense_53/bias/rms
?
-RMSprop/dense_53/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_53/bias/rms*
_output_shapes
:@*
dtype0
?
RMSprop/dense_54/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*,
shared_nameRMSprop/dense_54/kernel/rms
?
/RMSprop/dense_54/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_54/kernel/rms* 
_output_shapes
:
??*
dtype0
?
RMSprop/dense_54/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameRMSprop/dense_54/bias/rms
?
-RMSprop/dense_54/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_54/bias/rms*
_output_shapes
:*
dtype0

NoOpNoOp
?/
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?.
value?.B?. B?.
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer-4
layer_with_weights-4
layer-5
	optimizer
	variables
	regularization_losses

trainable_variables
	keras_api

signatures
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
 bias
!	variables
"trainable_variables
#regularization_losses
$	keras_api
R
%	variables
&trainable_variables
'regularization_losses
(	keras_api
h

)kernel
*bias
+	variables
,trainable_variables
-regularization_losses
.	keras_api
?
/iter
	0decay
1learning_rate
2momentum
3rho	rmsb	rmsc	rmsd	rmse	rmsf	rmsg	rmsh	 rmsi	)rmsj	*rmsk
F
0
1
2
3
4
5
6
 7
)8
*9
 
F
0
1
2
3
4
5
6
 7
)8
*9
?

4layers
	variables
5layer_metrics
	regularization_losses

trainable_variables
6layer_regularization_losses
7metrics
8non_trainable_variables
 
[Y
VARIABLE_VALUEdense_50/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_50/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?

9layers
	variables
:layer_metrics
trainable_variables
regularization_losses
;layer_regularization_losses
<metrics
=non_trainable_variables
[Y
VARIABLE_VALUEdense_51/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_51/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?

>layers
	variables
?layer_metrics
trainable_variables
regularization_losses
@layer_regularization_losses
Ametrics
Bnon_trainable_variables
[Y
VARIABLE_VALUEdense_52/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_52/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?

Clayers
	variables
Dlayer_metrics
trainable_variables
regularization_losses
Elayer_regularization_losses
Fmetrics
Gnon_trainable_variables
[Y
VARIABLE_VALUEdense_53/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_53/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

0
 1

0
 1
 
?

Hlayers
!	variables
Ilayer_metrics
"trainable_variables
#regularization_losses
Jlayer_regularization_losses
Kmetrics
Lnon_trainable_variables
 
 
 
?

Mlayers
%	variables
Nlayer_metrics
&trainable_variables
'regularization_losses
Olayer_regularization_losses
Pmetrics
Qnon_trainable_variables
[Y
VARIABLE_VALUEdense_54/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_54/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

)0
*1

)0
*1
 
?

Rlayers
+	variables
Slayer_metrics
,trainable_variables
-regularization_losses
Tlayer_regularization_losses
Umetrics
Vnon_trainable_variables
KI
VARIABLE_VALUERMSprop/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUERMSprop/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUERMSprop/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUERMSprop/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUERMSprop/rho(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUE
*
0
1
2
3
4
5
 
 

W0
X1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	Ytotal
	Zcount
[	variables
\	keras_api
D
	]total
	^count
_
_fn_kwargs
`	variables
a	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

Y0
Z1

[	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

]0
^1

`	variables
??
VARIABLE_VALUERMSprop/dense_50/kernel/rmsTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUERMSprop/dense_50/bias/rmsRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/dense_51/kernel/rmsTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUERMSprop/dense_51/bias/rmsRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/dense_52/kernel/rmsTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUERMSprop/dense_52/bias/rmsRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/dense_53/kernel/rmsTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUERMSprop/dense_53/bias/rmsRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/dense_54/kernel/rmsTlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUERMSprop/dense_54/bias/rmsRlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_dense_50_inputPlaceholder*/
_output_shapes
:?????????*
dtype0*$
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_dense_50_inputdense_50/kerneldense_50/biasdense_51/kerneldense_51/biasdense_52/kerneldense_52/biasdense_53/kerneldense_53/biasdense_54/kerneldense_54/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_14306
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_50/kernel/Read/ReadVariableOp!dense_50/bias/Read/ReadVariableOp#dense_51/kernel/Read/ReadVariableOp!dense_51/bias/Read/ReadVariableOp#dense_52/kernel/Read/ReadVariableOp!dense_52/bias/Read/ReadVariableOp#dense_53/kernel/Read/ReadVariableOp!dense_53/bias/Read/ReadVariableOp#dense_54/kernel/Read/ReadVariableOp!dense_54/bias/Read/ReadVariableOp RMSprop/iter/Read/ReadVariableOp!RMSprop/decay/Read/ReadVariableOp)RMSprop/learning_rate/Read/ReadVariableOp$RMSprop/momentum/Read/ReadVariableOpRMSprop/rho/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp/RMSprop/dense_50/kernel/rms/Read/ReadVariableOp-RMSprop/dense_50/bias/rms/Read/ReadVariableOp/RMSprop/dense_51/kernel/rms/Read/ReadVariableOp-RMSprop/dense_51/bias/rms/Read/ReadVariableOp/RMSprop/dense_52/kernel/rms/Read/ReadVariableOp-RMSprop/dense_52/bias/rms/Read/ReadVariableOp/RMSprop/dense_53/kernel/rms/Read/ReadVariableOp-RMSprop/dense_53/bias/rms/Read/ReadVariableOp/RMSprop/dense_54/kernel/rms/Read/ReadVariableOp-RMSprop/dense_54/bias/rms/Read/ReadVariableOpConst**
Tin#
!2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference__traced_save_14899
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_50/kerneldense_50/biasdense_51/kerneldense_51/biasdense_52/kerneldense_52/biasdense_53/kerneldense_53/biasdense_54/kerneldense_54/biasRMSprop/iterRMSprop/decayRMSprop/learning_rateRMSprop/momentumRMSprop/rhototalcounttotal_1count_1RMSprop/dense_50/kernel/rmsRMSprop/dense_50/bias/rmsRMSprop/dense_51/kernel/rmsRMSprop/dense_51/bias/rmsRMSprop/dense_52/kernel/rmsRMSprop/dense_52/bias/rmsRMSprop/dense_53/kernel/rmsRMSprop/dense_53/bias/rmsRMSprop/dense_54/kernel/rmsRMSprop/dense_54/bias/rms*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__traced_restore_14996??	
?
`
D__inference_flatten_3_layer_call_and_return_conditional_losses_14764

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????@n  2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:???????????2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
E
)__inference_flatten_3_layer_call_fn_14769

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_3_layer_call_and_return_conditional_losses_140092
PartitionedCalln
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
-__inference_sequential_10_layer_call_fn_14598

inputs
unknown:	?
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
	unknown_3:
??
	unknown_4:	?
	unknown_5:	?@
	unknown_6:@
	unknown_7:
??
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_10_layer_call_and_return_conditional_losses_141652
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?!
?
C__inference_dense_53_layer_call_and_return_conditional_losses_13997

inputs4
!tensordot_readvariableop_resource:	?@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	?@*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesu
Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*0
_output_shapes
:??????????2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*/
_output_shapes
:?????????@2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
C__inference_dense_54_layer_call_and_return_conditional_losses_14022

inputs2
matmul_readvariableop_resource:
??-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
-__inference_sequential_10_layer_call_fn_14573

inputs
unknown:	?
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
	unknown_3:
??
	unknown_4:	?
	unknown_5:	?@
	unknown_6:@
	unknown_7:
??
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_10_layer_call_and_return_conditional_losses_140292
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?!
?
C__inference_dense_52_layer_call_and_return_conditional_losses_13960

inputs5
!tensordot_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesu
Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*0
_output_shapes
:??????????2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*0
_output_shapes
:??????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?A
?
__inference__traced_save_14899
file_prefix.
*savev2_dense_50_kernel_read_readvariableop,
(savev2_dense_50_bias_read_readvariableop.
*savev2_dense_51_kernel_read_readvariableop,
(savev2_dense_51_bias_read_readvariableop.
*savev2_dense_52_kernel_read_readvariableop,
(savev2_dense_52_bias_read_readvariableop.
*savev2_dense_53_kernel_read_readvariableop,
(savev2_dense_53_bias_read_readvariableop.
*savev2_dense_54_kernel_read_readvariableop,
(savev2_dense_54_bias_read_readvariableop+
'savev2_rmsprop_iter_read_readvariableop	,
(savev2_rmsprop_decay_read_readvariableop4
0savev2_rmsprop_learning_rate_read_readvariableop/
+savev2_rmsprop_momentum_read_readvariableop*
&savev2_rmsprop_rho_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop:
6savev2_rmsprop_dense_50_kernel_rms_read_readvariableop8
4savev2_rmsprop_dense_50_bias_rms_read_readvariableop:
6savev2_rmsprop_dense_51_kernel_rms_read_readvariableop8
4savev2_rmsprop_dense_51_bias_rms_read_readvariableop:
6savev2_rmsprop_dense_52_kernel_rms_read_readvariableop8
4savev2_rmsprop_dense_52_bias_rms_read_readvariableop:
6savev2_rmsprop_dense_53_kernel_rms_read_readvariableop8
4savev2_rmsprop_dense_53_bias_rms_read_readvariableop:
6savev2_rmsprop_dense_54_kernel_rms_read_readvariableop8
4savev2_rmsprop_dense_54_bias_rms_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*O
valueFBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_50_kernel_read_readvariableop(savev2_dense_50_bias_read_readvariableop*savev2_dense_51_kernel_read_readvariableop(savev2_dense_51_bias_read_readvariableop*savev2_dense_52_kernel_read_readvariableop(savev2_dense_52_bias_read_readvariableop*savev2_dense_53_kernel_read_readvariableop(savev2_dense_53_bias_read_readvariableop*savev2_dense_54_kernel_read_readvariableop(savev2_dense_54_bias_read_readvariableop'savev2_rmsprop_iter_read_readvariableop(savev2_rmsprop_decay_read_readvariableop0savev2_rmsprop_learning_rate_read_readvariableop+savev2_rmsprop_momentum_read_readvariableop&savev2_rmsprop_rho_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop6savev2_rmsprop_dense_50_kernel_rms_read_readvariableop4savev2_rmsprop_dense_50_bias_rms_read_readvariableop6savev2_rmsprop_dense_51_kernel_rms_read_readvariableop4savev2_rmsprop_dense_51_bias_rms_read_readvariableop6savev2_rmsprop_dense_52_kernel_rms_read_readvariableop4savev2_rmsprop_dense_52_bias_rms_read_readvariableop6savev2_rmsprop_dense_53_kernel_rms_read_readvariableop4savev2_rmsprop_dense_53_bias_rms_read_readvariableop6savev2_rmsprop_dense_54_kernel_rms_read_readvariableop4savev2_rmsprop_dense_54_bias_rms_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *,
dtypes"
 2	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :	?:?:
??:?:
??:?:	?@:@:
??:: : : : : : : : : :	?:?:
??:?:
??:?:	?@:@:
??:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	?:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?@: 

_output_shapes
:@:&	"
 
_output_shapes
:
??: 


_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	?:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?@: 

_output_shapes
:@:&"
 
_output_shapes
:
??: 

_output_shapes
::

_output_shapes
: 
?
`
D__inference_flatten_3_layer_call_and_return_conditional_losses_14009

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????@n  2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:???????????2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?!
?
C__inference_dense_50_layer_call_and_return_conditional_losses_14629

inputs4
!tensordot_readvariableop_resource:	?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	?*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesu
Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*/
_output_shapes
:?????????2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*0
_output_shapes
:??????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
C__inference_dense_54_layer_call_and_return_conditional_losses_14780

inputs2
matmul_readvariableop_resource:
??-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
(__inference_dense_54_layer_call_fn_14789

inputs
unknown:
??
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_54_layer_call_and_return_conditional_losses_140222
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?!
?
C__inference_dense_51_layer_call_and_return_conditional_losses_14669

inputs5
!tensordot_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesu
Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*0
_output_shapes
:??????????2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*0
_output_shapes
:??????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
H__inference_sequential_10_layer_call_and_return_conditional_losses_14548

inputs=
*dense_50_tensordot_readvariableop_resource:	?7
(dense_50_biasadd_readvariableop_resource:	?>
*dense_51_tensordot_readvariableop_resource:
??7
(dense_51_biasadd_readvariableop_resource:	?>
*dense_52_tensordot_readvariableop_resource:
??7
(dense_52_biasadd_readvariableop_resource:	?=
*dense_53_tensordot_readvariableop_resource:	?@6
(dense_53_biasadd_readvariableop_resource:@;
'dense_54_matmul_readvariableop_resource:
??6
(dense_54_biasadd_readvariableop_resource:
identity??dense_50/BiasAdd/ReadVariableOp?!dense_50/Tensordot/ReadVariableOp?dense_51/BiasAdd/ReadVariableOp?!dense_51/Tensordot/ReadVariableOp?dense_52/BiasAdd/ReadVariableOp?!dense_52/Tensordot/ReadVariableOp?dense_53/BiasAdd/ReadVariableOp?!dense_53/Tensordot/ReadVariableOp?dense_54/BiasAdd/ReadVariableOp?dense_54/MatMul/ReadVariableOp?
!dense_50/Tensordot/ReadVariableOpReadVariableOp*dense_50_tensordot_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!dense_50/Tensordot/ReadVariableOp|
dense_50/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_50/Tensordot/axes?
dense_50/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
dense_50/Tensordot/freej
dense_50/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_50/Tensordot/Shape?
 dense_50/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_50/Tensordot/GatherV2/axis?
dense_50/Tensordot/GatherV2GatherV2!dense_50/Tensordot/Shape:output:0 dense_50/Tensordot/free:output:0)dense_50/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_50/Tensordot/GatherV2?
"dense_50/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_50/Tensordot/GatherV2_1/axis?
dense_50/Tensordot/GatherV2_1GatherV2!dense_50/Tensordot/Shape:output:0 dense_50/Tensordot/axes:output:0+dense_50/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_50/Tensordot/GatherV2_1~
dense_50/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_50/Tensordot/Const?
dense_50/Tensordot/ProdProd$dense_50/Tensordot/GatherV2:output:0!dense_50/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_50/Tensordot/Prod?
dense_50/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_50/Tensordot/Const_1?
dense_50/Tensordot/Prod_1Prod&dense_50/Tensordot/GatherV2_1:output:0#dense_50/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_50/Tensordot/Prod_1?
dense_50/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_50/Tensordot/concat/axis?
dense_50/Tensordot/concatConcatV2 dense_50/Tensordot/free:output:0 dense_50/Tensordot/axes:output:0'dense_50/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_50/Tensordot/concat?
dense_50/Tensordot/stackPack dense_50/Tensordot/Prod:output:0"dense_50/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_50/Tensordot/stack?
dense_50/Tensordot/transpose	Transposeinputs"dense_50/Tensordot/concat:output:0*
T0*/
_output_shapes
:?????????2
dense_50/Tensordot/transpose?
dense_50/Tensordot/ReshapeReshape dense_50/Tensordot/transpose:y:0!dense_50/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_50/Tensordot/Reshape?
dense_50/Tensordot/MatMulMatMul#dense_50/Tensordot/Reshape:output:0)dense_50/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_50/Tensordot/MatMul?
dense_50/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense_50/Tensordot/Const_2?
 dense_50/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_50/Tensordot/concat_1/axis?
dense_50/Tensordot/concat_1ConcatV2$dense_50/Tensordot/GatherV2:output:0#dense_50/Tensordot/Const_2:output:0)dense_50/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_50/Tensordot/concat_1?
dense_50/TensordotReshape#dense_50/Tensordot/MatMul:product:0$dense_50/Tensordot/concat_1:output:0*
T0*0
_output_shapes
:??????????2
dense_50/Tensordot?
dense_50/BiasAdd/ReadVariableOpReadVariableOp(dense_50_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_50/BiasAdd/ReadVariableOp?
dense_50/BiasAddBiasAdddense_50/Tensordot:output:0'dense_50/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
dense_50/BiasAdd|
dense_50/ReluReludense_50/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
dense_50/Relu?
!dense_51/Tensordot/ReadVariableOpReadVariableOp*dense_51_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype02#
!dense_51/Tensordot/ReadVariableOp|
dense_51/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_51/Tensordot/axes?
dense_51/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
dense_51/Tensordot/free
dense_51/Tensordot/ShapeShapedense_50/Relu:activations:0*
T0*
_output_shapes
:2
dense_51/Tensordot/Shape?
 dense_51/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_51/Tensordot/GatherV2/axis?
dense_51/Tensordot/GatherV2GatherV2!dense_51/Tensordot/Shape:output:0 dense_51/Tensordot/free:output:0)dense_51/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_51/Tensordot/GatherV2?
"dense_51/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_51/Tensordot/GatherV2_1/axis?
dense_51/Tensordot/GatherV2_1GatherV2!dense_51/Tensordot/Shape:output:0 dense_51/Tensordot/axes:output:0+dense_51/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_51/Tensordot/GatherV2_1~
dense_51/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_51/Tensordot/Const?
dense_51/Tensordot/ProdProd$dense_51/Tensordot/GatherV2:output:0!dense_51/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_51/Tensordot/Prod?
dense_51/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_51/Tensordot/Const_1?
dense_51/Tensordot/Prod_1Prod&dense_51/Tensordot/GatherV2_1:output:0#dense_51/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_51/Tensordot/Prod_1?
dense_51/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_51/Tensordot/concat/axis?
dense_51/Tensordot/concatConcatV2 dense_51/Tensordot/free:output:0 dense_51/Tensordot/axes:output:0'dense_51/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_51/Tensordot/concat?
dense_51/Tensordot/stackPack dense_51/Tensordot/Prod:output:0"dense_51/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_51/Tensordot/stack?
dense_51/Tensordot/transpose	Transposedense_50/Relu:activations:0"dense_51/Tensordot/concat:output:0*
T0*0
_output_shapes
:??????????2
dense_51/Tensordot/transpose?
dense_51/Tensordot/ReshapeReshape dense_51/Tensordot/transpose:y:0!dense_51/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_51/Tensordot/Reshape?
dense_51/Tensordot/MatMulMatMul#dense_51/Tensordot/Reshape:output:0)dense_51/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_51/Tensordot/MatMul?
dense_51/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense_51/Tensordot/Const_2?
 dense_51/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_51/Tensordot/concat_1/axis?
dense_51/Tensordot/concat_1ConcatV2$dense_51/Tensordot/GatherV2:output:0#dense_51/Tensordot/Const_2:output:0)dense_51/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_51/Tensordot/concat_1?
dense_51/TensordotReshape#dense_51/Tensordot/MatMul:product:0$dense_51/Tensordot/concat_1:output:0*
T0*0
_output_shapes
:??????????2
dense_51/Tensordot?
dense_51/BiasAdd/ReadVariableOpReadVariableOp(dense_51_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_51/BiasAdd/ReadVariableOp?
dense_51/BiasAddBiasAdddense_51/Tensordot:output:0'dense_51/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
dense_51/BiasAdd|
dense_51/ReluReludense_51/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
dense_51/Relu?
!dense_52/Tensordot/ReadVariableOpReadVariableOp*dense_52_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype02#
!dense_52/Tensordot/ReadVariableOp|
dense_52/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_52/Tensordot/axes?
dense_52/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
dense_52/Tensordot/free
dense_52/Tensordot/ShapeShapedense_51/Relu:activations:0*
T0*
_output_shapes
:2
dense_52/Tensordot/Shape?
 dense_52/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_52/Tensordot/GatherV2/axis?
dense_52/Tensordot/GatherV2GatherV2!dense_52/Tensordot/Shape:output:0 dense_52/Tensordot/free:output:0)dense_52/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_52/Tensordot/GatherV2?
"dense_52/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_52/Tensordot/GatherV2_1/axis?
dense_52/Tensordot/GatherV2_1GatherV2!dense_52/Tensordot/Shape:output:0 dense_52/Tensordot/axes:output:0+dense_52/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_52/Tensordot/GatherV2_1~
dense_52/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_52/Tensordot/Const?
dense_52/Tensordot/ProdProd$dense_52/Tensordot/GatherV2:output:0!dense_52/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_52/Tensordot/Prod?
dense_52/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_52/Tensordot/Const_1?
dense_52/Tensordot/Prod_1Prod&dense_52/Tensordot/GatherV2_1:output:0#dense_52/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_52/Tensordot/Prod_1?
dense_52/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_52/Tensordot/concat/axis?
dense_52/Tensordot/concatConcatV2 dense_52/Tensordot/free:output:0 dense_52/Tensordot/axes:output:0'dense_52/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_52/Tensordot/concat?
dense_52/Tensordot/stackPack dense_52/Tensordot/Prod:output:0"dense_52/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_52/Tensordot/stack?
dense_52/Tensordot/transpose	Transposedense_51/Relu:activations:0"dense_52/Tensordot/concat:output:0*
T0*0
_output_shapes
:??????????2
dense_52/Tensordot/transpose?
dense_52/Tensordot/ReshapeReshape dense_52/Tensordot/transpose:y:0!dense_52/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_52/Tensordot/Reshape?
dense_52/Tensordot/MatMulMatMul#dense_52/Tensordot/Reshape:output:0)dense_52/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_52/Tensordot/MatMul?
dense_52/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense_52/Tensordot/Const_2?
 dense_52/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_52/Tensordot/concat_1/axis?
dense_52/Tensordot/concat_1ConcatV2$dense_52/Tensordot/GatherV2:output:0#dense_52/Tensordot/Const_2:output:0)dense_52/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_52/Tensordot/concat_1?
dense_52/TensordotReshape#dense_52/Tensordot/MatMul:product:0$dense_52/Tensordot/concat_1:output:0*
T0*0
_output_shapes
:??????????2
dense_52/Tensordot?
dense_52/BiasAdd/ReadVariableOpReadVariableOp(dense_52_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_52/BiasAdd/ReadVariableOp?
dense_52/BiasAddBiasAdddense_52/Tensordot:output:0'dense_52/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
dense_52/BiasAdd|
dense_52/ReluReludense_52/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
dense_52/Relu?
!dense_53/Tensordot/ReadVariableOpReadVariableOp*dense_53_tensordot_readvariableop_resource*
_output_shapes
:	?@*
dtype02#
!dense_53/Tensordot/ReadVariableOp|
dense_53/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_53/Tensordot/axes?
dense_53/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
dense_53/Tensordot/free
dense_53/Tensordot/ShapeShapedense_52/Relu:activations:0*
T0*
_output_shapes
:2
dense_53/Tensordot/Shape?
 dense_53/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_53/Tensordot/GatherV2/axis?
dense_53/Tensordot/GatherV2GatherV2!dense_53/Tensordot/Shape:output:0 dense_53/Tensordot/free:output:0)dense_53/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_53/Tensordot/GatherV2?
"dense_53/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_53/Tensordot/GatherV2_1/axis?
dense_53/Tensordot/GatherV2_1GatherV2!dense_53/Tensordot/Shape:output:0 dense_53/Tensordot/axes:output:0+dense_53/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_53/Tensordot/GatherV2_1~
dense_53/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_53/Tensordot/Const?
dense_53/Tensordot/ProdProd$dense_53/Tensordot/GatherV2:output:0!dense_53/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_53/Tensordot/Prod?
dense_53/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_53/Tensordot/Const_1?
dense_53/Tensordot/Prod_1Prod&dense_53/Tensordot/GatherV2_1:output:0#dense_53/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_53/Tensordot/Prod_1?
dense_53/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_53/Tensordot/concat/axis?
dense_53/Tensordot/concatConcatV2 dense_53/Tensordot/free:output:0 dense_53/Tensordot/axes:output:0'dense_53/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_53/Tensordot/concat?
dense_53/Tensordot/stackPack dense_53/Tensordot/Prod:output:0"dense_53/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_53/Tensordot/stack?
dense_53/Tensordot/transpose	Transposedense_52/Relu:activations:0"dense_53/Tensordot/concat:output:0*
T0*0
_output_shapes
:??????????2
dense_53/Tensordot/transpose?
dense_53/Tensordot/ReshapeReshape dense_53/Tensordot/transpose:y:0!dense_53/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_53/Tensordot/Reshape?
dense_53/Tensordot/MatMulMatMul#dense_53/Tensordot/Reshape:output:0)dense_53/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_53/Tensordot/MatMul?
dense_53/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
dense_53/Tensordot/Const_2?
 dense_53/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_53/Tensordot/concat_1/axis?
dense_53/Tensordot/concat_1ConcatV2$dense_53/Tensordot/GatherV2:output:0#dense_53/Tensordot/Const_2:output:0)dense_53/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_53/Tensordot/concat_1?
dense_53/TensordotReshape#dense_53/Tensordot/MatMul:product:0$dense_53/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:?????????@2
dense_53/Tensordot?
dense_53/BiasAdd/ReadVariableOpReadVariableOp(dense_53_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_53/BiasAdd/ReadVariableOp?
dense_53/BiasAddBiasAdddense_53/Tensordot:output:0'dense_53/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
dense_53/BiasAdd{
dense_53/ReluReludense_53/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
dense_53/Relus
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"????@n  2
flatten_3/Const?
flatten_3/ReshapeReshapedense_53/Relu:activations:0flatten_3/Const:output:0*
T0*)
_output_shapes
:???????????2
flatten_3/Reshape?
dense_54/MatMul/ReadVariableOpReadVariableOp'dense_54_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_54/MatMul/ReadVariableOp?
dense_54/MatMulMatMulflatten_3/Reshape:output:0&dense_54/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_54/MatMul?
dense_54/BiasAdd/ReadVariableOpReadVariableOp(dense_54_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_54/BiasAdd/ReadVariableOp?
dense_54/BiasAddBiasAdddense_54/MatMul:product:0'dense_54/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_54/BiasAdd|
dense_54/SoftmaxSoftmaxdense_54/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_54/Softmax?
IdentityIdentitydense_54/Softmax:softmax:0 ^dense_50/BiasAdd/ReadVariableOp"^dense_50/Tensordot/ReadVariableOp ^dense_51/BiasAdd/ReadVariableOp"^dense_51/Tensordot/ReadVariableOp ^dense_52/BiasAdd/ReadVariableOp"^dense_52/Tensordot/ReadVariableOp ^dense_53/BiasAdd/ReadVariableOp"^dense_53/Tensordot/ReadVariableOp ^dense_54/BiasAdd/ReadVariableOp^dense_54/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : 2B
dense_50/BiasAdd/ReadVariableOpdense_50/BiasAdd/ReadVariableOp2F
!dense_50/Tensordot/ReadVariableOp!dense_50/Tensordot/ReadVariableOp2B
dense_51/BiasAdd/ReadVariableOpdense_51/BiasAdd/ReadVariableOp2F
!dense_51/Tensordot/ReadVariableOp!dense_51/Tensordot/ReadVariableOp2B
dense_52/BiasAdd/ReadVariableOpdense_52/BiasAdd/ReadVariableOp2F
!dense_52/Tensordot/ReadVariableOp!dense_52/Tensordot/ReadVariableOp2B
dense_53/BiasAdd/ReadVariableOpdense_53/BiasAdd/ReadVariableOp2F
!dense_53/Tensordot/ReadVariableOp!dense_53/Tensordot/ReadVariableOp2B
dense_54/BiasAdd/ReadVariableOpdense_54/BiasAdd/ReadVariableOp2@
dense_54/MatMul/ReadVariableOpdense_54/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
(__inference_dense_51_layer_call_fn_14678

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_51_layer_call_and_return_conditional_losses_139232
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?}
?
!__inference__traced_restore_14996
file_prefix3
 assignvariableop_dense_50_kernel:	?/
 assignvariableop_1_dense_50_bias:	?6
"assignvariableop_2_dense_51_kernel:
??/
 assignvariableop_3_dense_51_bias:	?6
"assignvariableop_4_dense_52_kernel:
??/
 assignvariableop_5_dense_52_bias:	?5
"assignvariableop_6_dense_53_kernel:	?@.
 assignvariableop_7_dense_53_bias:@6
"assignvariableop_8_dense_54_kernel:
??.
 assignvariableop_9_dense_54_bias:*
 assignvariableop_10_rmsprop_iter:	 +
!assignvariableop_11_rmsprop_decay: 3
)assignvariableop_12_rmsprop_learning_rate: .
$assignvariableop_13_rmsprop_momentum: )
assignvariableop_14_rmsprop_rho: #
assignvariableop_15_total: #
assignvariableop_16_count: %
assignvariableop_17_total_1: %
assignvariableop_18_count_1: B
/assignvariableop_19_rmsprop_dense_50_kernel_rms:	?<
-assignvariableop_20_rmsprop_dense_50_bias_rms:	?C
/assignvariableop_21_rmsprop_dense_51_kernel_rms:
??<
-assignvariableop_22_rmsprop_dense_51_bias_rms:	?C
/assignvariableop_23_rmsprop_dense_52_kernel_rms:
??<
-assignvariableop_24_rmsprop_dense_52_bias_rms:	?B
/assignvariableop_25_rmsprop_dense_53_kernel_rms:	?@;
-assignvariableop_26_rmsprop_dense_53_bias_rms:@C
/assignvariableop_27_rmsprop_dense_54_kernel_rms:
??;
-assignvariableop_28_rmsprop_dense_54_bias_rms:
identity_30??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*O
valueFBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapesz
x::::::::::::::::::::::::::::::*,
dtypes"
 2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp assignvariableop_dense_50_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_50_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_51_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_51_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_52_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_52_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_53_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_53_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp"assignvariableop_8_dense_54_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp assignvariableop_9_dense_54_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp assignvariableop_10_rmsprop_iterIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp!assignvariableop_11_rmsprop_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp)assignvariableop_12_rmsprop_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp$assignvariableop_13_rmsprop_momentumIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_rmsprop_rhoIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_totalIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_countIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_total_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpassignvariableop_18_count_1Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp/assignvariableop_19_rmsprop_dense_50_kernel_rmsIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp-assignvariableop_20_rmsprop_dense_50_bias_rmsIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp/assignvariableop_21_rmsprop_dense_51_kernel_rmsIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp-assignvariableop_22_rmsprop_dense_51_bias_rmsIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp/assignvariableop_23_rmsprop_dense_52_kernel_rmsIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp-assignvariableop_24_rmsprop_dense_52_bias_rmsIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp/assignvariableop_25_rmsprop_dense_53_kernel_rmsIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp-assignvariableop_26_rmsprop_dense_53_bias_rmsIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp/assignvariableop_27_rmsprop_dense_54_kernel_rmsIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp-assignvariableop_28_rmsprop_dense_54_bias_rmsIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_289
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_29Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_29?
Identity_30IdentityIdentity_29:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_30"#
identity_30Identity_30:output:0*O
_input_shapes>
<: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?!
?
C__inference_dense_52_layer_call_and_return_conditional_losses_14709

inputs5
!tensordot_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesu
Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*0
_output_shapes
:??????????2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*0
_output_shapes
:??????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?!
?
C__inference_dense_50_layer_call_and_return_conditional_losses_13886

inputs4
!tensordot_readvariableop_resource:	?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	?*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesu
Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*/
_output_shapes
:?????????2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*0
_output_shapes
:??????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
H__inference_sequential_10_layer_call_and_return_conditional_losses_14427

inputs=
*dense_50_tensordot_readvariableop_resource:	?7
(dense_50_biasadd_readvariableop_resource:	?>
*dense_51_tensordot_readvariableop_resource:
??7
(dense_51_biasadd_readvariableop_resource:	?>
*dense_52_tensordot_readvariableop_resource:
??7
(dense_52_biasadd_readvariableop_resource:	?=
*dense_53_tensordot_readvariableop_resource:	?@6
(dense_53_biasadd_readvariableop_resource:@;
'dense_54_matmul_readvariableop_resource:
??6
(dense_54_biasadd_readvariableop_resource:
identity??dense_50/BiasAdd/ReadVariableOp?!dense_50/Tensordot/ReadVariableOp?dense_51/BiasAdd/ReadVariableOp?!dense_51/Tensordot/ReadVariableOp?dense_52/BiasAdd/ReadVariableOp?!dense_52/Tensordot/ReadVariableOp?dense_53/BiasAdd/ReadVariableOp?!dense_53/Tensordot/ReadVariableOp?dense_54/BiasAdd/ReadVariableOp?dense_54/MatMul/ReadVariableOp?
!dense_50/Tensordot/ReadVariableOpReadVariableOp*dense_50_tensordot_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!dense_50/Tensordot/ReadVariableOp|
dense_50/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_50/Tensordot/axes?
dense_50/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
dense_50/Tensordot/freej
dense_50/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_50/Tensordot/Shape?
 dense_50/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_50/Tensordot/GatherV2/axis?
dense_50/Tensordot/GatherV2GatherV2!dense_50/Tensordot/Shape:output:0 dense_50/Tensordot/free:output:0)dense_50/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_50/Tensordot/GatherV2?
"dense_50/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_50/Tensordot/GatherV2_1/axis?
dense_50/Tensordot/GatherV2_1GatherV2!dense_50/Tensordot/Shape:output:0 dense_50/Tensordot/axes:output:0+dense_50/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_50/Tensordot/GatherV2_1~
dense_50/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_50/Tensordot/Const?
dense_50/Tensordot/ProdProd$dense_50/Tensordot/GatherV2:output:0!dense_50/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_50/Tensordot/Prod?
dense_50/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_50/Tensordot/Const_1?
dense_50/Tensordot/Prod_1Prod&dense_50/Tensordot/GatherV2_1:output:0#dense_50/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_50/Tensordot/Prod_1?
dense_50/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_50/Tensordot/concat/axis?
dense_50/Tensordot/concatConcatV2 dense_50/Tensordot/free:output:0 dense_50/Tensordot/axes:output:0'dense_50/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_50/Tensordot/concat?
dense_50/Tensordot/stackPack dense_50/Tensordot/Prod:output:0"dense_50/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_50/Tensordot/stack?
dense_50/Tensordot/transpose	Transposeinputs"dense_50/Tensordot/concat:output:0*
T0*/
_output_shapes
:?????????2
dense_50/Tensordot/transpose?
dense_50/Tensordot/ReshapeReshape dense_50/Tensordot/transpose:y:0!dense_50/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_50/Tensordot/Reshape?
dense_50/Tensordot/MatMulMatMul#dense_50/Tensordot/Reshape:output:0)dense_50/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_50/Tensordot/MatMul?
dense_50/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense_50/Tensordot/Const_2?
 dense_50/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_50/Tensordot/concat_1/axis?
dense_50/Tensordot/concat_1ConcatV2$dense_50/Tensordot/GatherV2:output:0#dense_50/Tensordot/Const_2:output:0)dense_50/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_50/Tensordot/concat_1?
dense_50/TensordotReshape#dense_50/Tensordot/MatMul:product:0$dense_50/Tensordot/concat_1:output:0*
T0*0
_output_shapes
:??????????2
dense_50/Tensordot?
dense_50/BiasAdd/ReadVariableOpReadVariableOp(dense_50_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_50/BiasAdd/ReadVariableOp?
dense_50/BiasAddBiasAdddense_50/Tensordot:output:0'dense_50/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
dense_50/BiasAdd|
dense_50/ReluReludense_50/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
dense_50/Relu?
!dense_51/Tensordot/ReadVariableOpReadVariableOp*dense_51_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype02#
!dense_51/Tensordot/ReadVariableOp|
dense_51/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_51/Tensordot/axes?
dense_51/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
dense_51/Tensordot/free
dense_51/Tensordot/ShapeShapedense_50/Relu:activations:0*
T0*
_output_shapes
:2
dense_51/Tensordot/Shape?
 dense_51/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_51/Tensordot/GatherV2/axis?
dense_51/Tensordot/GatherV2GatherV2!dense_51/Tensordot/Shape:output:0 dense_51/Tensordot/free:output:0)dense_51/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_51/Tensordot/GatherV2?
"dense_51/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_51/Tensordot/GatherV2_1/axis?
dense_51/Tensordot/GatherV2_1GatherV2!dense_51/Tensordot/Shape:output:0 dense_51/Tensordot/axes:output:0+dense_51/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_51/Tensordot/GatherV2_1~
dense_51/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_51/Tensordot/Const?
dense_51/Tensordot/ProdProd$dense_51/Tensordot/GatherV2:output:0!dense_51/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_51/Tensordot/Prod?
dense_51/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_51/Tensordot/Const_1?
dense_51/Tensordot/Prod_1Prod&dense_51/Tensordot/GatherV2_1:output:0#dense_51/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_51/Tensordot/Prod_1?
dense_51/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_51/Tensordot/concat/axis?
dense_51/Tensordot/concatConcatV2 dense_51/Tensordot/free:output:0 dense_51/Tensordot/axes:output:0'dense_51/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_51/Tensordot/concat?
dense_51/Tensordot/stackPack dense_51/Tensordot/Prod:output:0"dense_51/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_51/Tensordot/stack?
dense_51/Tensordot/transpose	Transposedense_50/Relu:activations:0"dense_51/Tensordot/concat:output:0*
T0*0
_output_shapes
:??????????2
dense_51/Tensordot/transpose?
dense_51/Tensordot/ReshapeReshape dense_51/Tensordot/transpose:y:0!dense_51/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_51/Tensordot/Reshape?
dense_51/Tensordot/MatMulMatMul#dense_51/Tensordot/Reshape:output:0)dense_51/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_51/Tensordot/MatMul?
dense_51/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense_51/Tensordot/Const_2?
 dense_51/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_51/Tensordot/concat_1/axis?
dense_51/Tensordot/concat_1ConcatV2$dense_51/Tensordot/GatherV2:output:0#dense_51/Tensordot/Const_2:output:0)dense_51/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_51/Tensordot/concat_1?
dense_51/TensordotReshape#dense_51/Tensordot/MatMul:product:0$dense_51/Tensordot/concat_1:output:0*
T0*0
_output_shapes
:??????????2
dense_51/Tensordot?
dense_51/BiasAdd/ReadVariableOpReadVariableOp(dense_51_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_51/BiasAdd/ReadVariableOp?
dense_51/BiasAddBiasAdddense_51/Tensordot:output:0'dense_51/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
dense_51/BiasAdd|
dense_51/ReluReludense_51/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
dense_51/Relu?
!dense_52/Tensordot/ReadVariableOpReadVariableOp*dense_52_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype02#
!dense_52/Tensordot/ReadVariableOp|
dense_52/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_52/Tensordot/axes?
dense_52/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
dense_52/Tensordot/free
dense_52/Tensordot/ShapeShapedense_51/Relu:activations:0*
T0*
_output_shapes
:2
dense_52/Tensordot/Shape?
 dense_52/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_52/Tensordot/GatherV2/axis?
dense_52/Tensordot/GatherV2GatherV2!dense_52/Tensordot/Shape:output:0 dense_52/Tensordot/free:output:0)dense_52/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_52/Tensordot/GatherV2?
"dense_52/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_52/Tensordot/GatherV2_1/axis?
dense_52/Tensordot/GatherV2_1GatherV2!dense_52/Tensordot/Shape:output:0 dense_52/Tensordot/axes:output:0+dense_52/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_52/Tensordot/GatherV2_1~
dense_52/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_52/Tensordot/Const?
dense_52/Tensordot/ProdProd$dense_52/Tensordot/GatherV2:output:0!dense_52/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_52/Tensordot/Prod?
dense_52/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_52/Tensordot/Const_1?
dense_52/Tensordot/Prod_1Prod&dense_52/Tensordot/GatherV2_1:output:0#dense_52/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_52/Tensordot/Prod_1?
dense_52/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_52/Tensordot/concat/axis?
dense_52/Tensordot/concatConcatV2 dense_52/Tensordot/free:output:0 dense_52/Tensordot/axes:output:0'dense_52/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_52/Tensordot/concat?
dense_52/Tensordot/stackPack dense_52/Tensordot/Prod:output:0"dense_52/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_52/Tensordot/stack?
dense_52/Tensordot/transpose	Transposedense_51/Relu:activations:0"dense_52/Tensordot/concat:output:0*
T0*0
_output_shapes
:??????????2
dense_52/Tensordot/transpose?
dense_52/Tensordot/ReshapeReshape dense_52/Tensordot/transpose:y:0!dense_52/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_52/Tensordot/Reshape?
dense_52/Tensordot/MatMulMatMul#dense_52/Tensordot/Reshape:output:0)dense_52/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_52/Tensordot/MatMul?
dense_52/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense_52/Tensordot/Const_2?
 dense_52/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_52/Tensordot/concat_1/axis?
dense_52/Tensordot/concat_1ConcatV2$dense_52/Tensordot/GatherV2:output:0#dense_52/Tensordot/Const_2:output:0)dense_52/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_52/Tensordot/concat_1?
dense_52/TensordotReshape#dense_52/Tensordot/MatMul:product:0$dense_52/Tensordot/concat_1:output:0*
T0*0
_output_shapes
:??????????2
dense_52/Tensordot?
dense_52/BiasAdd/ReadVariableOpReadVariableOp(dense_52_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_52/BiasAdd/ReadVariableOp?
dense_52/BiasAddBiasAdddense_52/Tensordot:output:0'dense_52/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
dense_52/BiasAdd|
dense_52/ReluReludense_52/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
dense_52/Relu?
!dense_53/Tensordot/ReadVariableOpReadVariableOp*dense_53_tensordot_readvariableop_resource*
_output_shapes
:	?@*
dtype02#
!dense_53/Tensordot/ReadVariableOp|
dense_53/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_53/Tensordot/axes?
dense_53/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
dense_53/Tensordot/free
dense_53/Tensordot/ShapeShapedense_52/Relu:activations:0*
T0*
_output_shapes
:2
dense_53/Tensordot/Shape?
 dense_53/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_53/Tensordot/GatherV2/axis?
dense_53/Tensordot/GatherV2GatherV2!dense_53/Tensordot/Shape:output:0 dense_53/Tensordot/free:output:0)dense_53/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_53/Tensordot/GatherV2?
"dense_53/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_53/Tensordot/GatherV2_1/axis?
dense_53/Tensordot/GatherV2_1GatherV2!dense_53/Tensordot/Shape:output:0 dense_53/Tensordot/axes:output:0+dense_53/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_53/Tensordot/GatherV2_1~
dense_53/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_53/Tensordot/Const?
dense_53/Tensordot/ProdProd$dense_53/Tensordot/GatherV2:output:0!dense_53/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_53/Tensordot/Prod?
dense_53/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_53/Tensordot/Const_1?
dense_53/Tensordot/Prod_1Prod&dense_53/Tensordot/GatherV2_1:output:0#dense_53/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_53/Tensordot/Prod_1?
dense_53/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_53/Tensordot/concat/axis?
dense_53/Tensordot/concatConcatV2 dense_53/Tensordot/free:output:0 dense_53/Tensordot/axes:output:0'dense_53/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_53/Tensordot/concat?
dense_53/Tensordot/stackPack dense_53/Tensordot/Prod:output:0"dense_53/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_53/Tensordot/stack?
dense_53/Tensordot/transpose	Transposedense_52/Relu:activations:0"dense_53/Tensordot/concat:output:0*
T0*0
_output_shapes
:??????????2
dense_53/Tensordot/transpose?
dense_53/Tensordot/ReshapeReshape dense_53/Tensordot/transpose:y:0!dense_53/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_53/Tensordot/Reshape?
dense_53/Tensordot/MatMulMatMul#dense_53/Tensordot/Reshape:output:0)dense_53/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_53/Tensordot/MatMul?
dense_53/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
dense_53/Tensordot/Const_2?
 dense_53/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_53/Tensordot/concat_1/axis?
dense_53/Tensordot/concat_1ConcatV2$dense_53/Tensordot/GatherV2:output:0#dense_53/Tensordot/Const_2:output:0)dense_53/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_53/Tensordot/concat_1?
dense_53/TensordotReshape#dense_53/Tensordot/MatMul:product:0$dense_53/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:?????????@2
dense_53/Tensordot?
dense_53/BiasAdd/ReadVariableOpReadVariableOp(dense_53_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_53/BiasAdd/ReadVariableOp?
dense_53/BiasAddBiasAdddense_53/Tensordot:output:0'dense_53/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
dense_53/BiasAdd{
dense_53/ReluReludense_53/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
dense_53/Relus
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"????@n  2
flatten_3/Const?
flatten_3/ReshapeReshapedense_53/Relu:activations:0flatten_3/Const:output:0*
T0*)
_output_shapes
:???????????2
flatten_3/Reshape?
dense_54/MatMul/ReadVariableOpReadVariableOp'dense_54_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_54/MatMul/ReadVariableOp?
dense_54/MatMulMatMulflatten_3/Reshape:output:0&dense_54/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_54/MatMul?
dense_54/BiasAdd/ReadVariableOpReadVariableOp(dense_54_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_54/BiasAdd/ReadVariableOp?
dense_54/BiasAddBiasAdddense_54/MatMul:product:0'dense_54/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_54/BiasAdd|
dense_54/SoftmaxSoftmaxdense_54/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_54/Softmax?
IdentityIdentitydense_54/Softmax:softmax:0 ^dense_50/BiasAdd/ReadVariableOp"^dense_50/Tensordot/ReadVariableOp ^dense_51/BiasAdd/ReadVariableOp"^dense_51/Tensordot/ReadVariableOp ^dense_52/BiasAdd/ReadVariableOp"^dense_52/Tensordot/ReadVariableOp ^dense_53/BiasAdd/ReadVariableOp"^dense_53/Tensordot/ReadVariableOp ^dense_54/BiasAdd/ReadVariableOp^dense_54/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : 2B
dense_50/BiasAdd/ReadVariableOpdense_50/BiasAdd/ReadVariableOp2F
!dense_50/Tensordot/ReadVariableOp!dense_50/Tensordot/ReadVariableOp2B
dense_51/BiasAdd/ReadVariableOpdense_51/BiasAdd/ReadVariableOp2F
!dense_51/Tensordot/ReadVariableOp!dense_51/Tensordot/ReadVariableOp2B
dense_52/BiasAdd/ReadVariableOpdense_52/BiasAdd/ReadVariableOp2F
!dense_52/Tensordot/ReadVariableOp!dense_52/Tensordot/ReadVariableOp2B
dense_53/BiasAdd/ReadVariableOpdense_53/BiasAdd/ReadVariableOp2F
!dense_53/Tensordot/ReadVariableOp!dense_53/Tensordot/ReadVariableOp2B
dense_54/BiasAdd/ReadVariableOpdense_54/BiasAdd/ReadVariableOp2@
dense_54/MatMul/ReadVariableOpdense_54/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?!
?
C__inference_dense_53_layer_call_and_return_conditional_losses_14749

inputs4
!tensordot_readvariableop_resource:	?@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	?@*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesu
Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*0
_output_shapes
:??????????2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*/
_output_shapes
:?????????@2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
H__inference_sequential_10_layer_call_and_return_conditional_losses_14273
dense_50_input!
dense_50_14246:	?
dense_50_14248:	?"
dense_51_14251:
??
dense_51_14253:	?"
dense_52_14256:
??
dense_52_14258:	?!
dense_53_14261:	?@
dense_53_14263:@"
dense_54_14267:
??
dense_54_14269:
identity?? dense_50/StatefulPartitionedCall? dense_51/StatefulPartitionedCall? dense_52/StatefulPartitionedCall? dense_53/StatefulPartitionedCall? dense_54/StatefulPartitionedCall?
 dense_50/StatefulPartitionedCallStatefulPartitionedCalldense_50_inputdense_50_14246dense_50_14248*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_50_layer_call_and_return_conditional_losses_138862"
 dense_50/StatefulPartitionedCall?
 dense_51/StatefulPartitionedCallStatefulPartitionedCall)dense_50/StatefulPartitionedCall:output:0dense_51_14251dense_51_14253*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_51_layer_call_and_return_conditional_losses_139232"
 dense_51/StatefulPartitionedCall?
 dense_52/StatefulPartitionedCallStatefulPartitionedCall)dense_51/StatefulPartitionedCall:output:0dense_52_14256dense_52_14258*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_52_layer_call_and_return_conditional_losses_139602"
 dense_52/StatefulPartitionedCall?
 dense_53/StatefulPartitionedCallStatefulPartitionedCall)dense_52/StatefulPartitionedCall:output:0dense_53_14261dense_53_14263*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_53_layer_call_and_return_conditional_losses_139972"
 dense_53/StatefulPartitionedCall?
flatten_3/PartitionedCallPartitionedCall)dense_53/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_3_layer_call_and_return_conditional_losses_140092
flatten_3/PartitionedCall?
 dense_54/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_54_14267dense_54_14269*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_54_layer_call_and_return_conditional_losses_140222"
 dense_54/StatefulPartitionedCall?
IdentityIdentity)dense_54/StatefulPartitionedCall:output:0!^dense_50/StatefulPartitionedCall!^dense_51/StatefulPartitionedCall!^dense_52/StatefulPartitionedCall!^dense_53/StatefulPartitionedCall!^dense_54/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : 2D
 dense_50/StatefulPartitionedCall dense_50/StatefulPartitionedCall2D
 dense_51/StatefulPartitionedCall dense_51/StatefulPartitionedCall2D
 dense_52/StatefulPartitionedCall dense_52/StatefulPartitionedCall2D
 dense_53/StatefulPartitionedCall dense_53/StatefulPartitionedCall2D
 dense_54/StatefulPartitionedCall dense_54/StatefulPartitionedCall:_ [
/
_output_shapes
:?????????
(
_user_specified_namedense_50_input
?	
?
#__inference_signature_wrapper_14306
dense_50_input
unknown:	?
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
	unknown_3:
??
	unknown_4:	?
	unknown_5:	?@
	unknown_6:@
	unknown_7:
??
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_50_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_138482
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:?????????
(
_user_specified_namedense_50_input
?
?
(__inference_dense_52_layer_call_fn_14718

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_52_layer_call_and_return_conditional_losses_139602
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
-__inference_sequential_10_layer_call_fn_14213
dense_50_input
unknown:	?
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
	unknown_3:
??
	unknown_4:	?
	unknown_5:	?@
	unknown_6:@
	unknown_7:
??
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_50_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_10_layer_call_and_return_conditional_losses_141652
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:?????????
(
_user_specified_namedense_50_input
?

?
-__inference_sequential_10_layer_call_fn_14052
dense_50_input
unknown:	?
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
	unknown_3:
??
	unknown_4:	?
	unknown_5:	?@
	unknown_6:@
	unknown_7:
??
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_50_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_10_layer_call_and_return_conditional_losses_140292
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:?????????
(
_user_specified_namedense_50_input
?
?
H__inference_sequential_10_layer_call_and_return_conditional_losses_14029

inputs!
dense_50_13887:	?
dense_50_13889:	?"
dense_51_13924:
??
dense_51_13926:	?"
dense_52_13961:
??
dense_52_13963:	?!
dense_53_13998:	?@
dense_53_14000:@"
dense_54_14023:
??
dense_54_14025:
identity?? dense_50/StatefulPartitionedCall? dense_51/StatefulPartitionedCall? dense_52/StatefulPartitionedCall? dense_53/StatefulPartitionedCall? dense_54/StatefulPartitionedCall?
 dense_50/StatefulPartitionedCallStatefulPartitionedCallinputsdense_50_13887dense_50_13889*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_50_layer_call_and_return_conditional_losses_138862"
 dense_50/StatefulPartitionedCall?
 dense_51/StatefulPartitionedCallStatefulPartitionedCall)dense_50/StatefulPartitionedCall:output:0dense_51_13924dense_51_13926*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_51_layer_call_and_return_conditional_losses_139232"
 dense_51/StatefulPartitionedCall?
 dense_52/StatefulPartitionedCallStatefulPartitionedCall)dense_51/StatefulPartitionedCall:output:0dense_52_13961dense_52_13963*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_52_layer_call_and_return_conditional_losses_139602"
 dense_52/StatefulPartitionedCall?
 dense_53/StatefulPartitionedCallStatefulPartitionedCall)dense_52/StatefulPartitionedCall:output:0dense_53_13998dense_53_14000*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_53_layer_call_and_return_conditional_losses_139972"
 dense_53/StatefulPartitionedCall?
flatten_3/PartitionedCallPartitionedCall)dense_53/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_3_layer_call_and_return_conditional_losses_140092
flatten_3/PartitionedCall?
 dense_54/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_54_14023dense_54_14025*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_54_layer_call_and_return_conditional_losses_140222"
 dense_54/StatefulPartitionedCall?
IdentityIdentity)dense_54/StatefulPartitionedCall:output:0!^dense_50/StatefulPartitionedCall!^dense_51/StatefulPartitionedCall!^dense_52/StatefulPartitionedCall!^dense_53/StatefulPartitionedCall!^dense_54/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : 2D
 dense_50/StatefulPartitionedCall dense_50/StatefulPartitionedCall2D
 dense_51/StatefulPartitionedCall dense_51/StatefulPartitionedCall2D
 dense_52/StatefulPartitionedCall dense_52/StatefulPartitionedCall2D
 dense_53/StatefulPartitionedCall dense_53/StatefulPartitionedCall2D
 dense_54/StatefulPartitionedCall dense_54/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
H__inference_sequential_10_layer_call_and_return_conditional_losses_14165

inputs!
dense_50_14138:	?
dense_50_14140:	?"
dense_51_14143:
??
dense_51_14145:	?"
dense_52_14148:
??
dense_52_14150:	?!
dense_53_14153:	?@
dense_53_14155:@"
dense_54_14159:
??
dense_54_14161:
identity?? dense_50/StatefulPartitionedCall? dense_51/StatefulPartitionedCall? dense_52/StatefulPartitionedCall? dense_53/StatefulPartitionedCall? dense_54/StatefulPartitionedCall?
 dense_50/StatefulPartitionedCallStatefulPartitionedCallinputsdense_50_14138dense_50_14140*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_50_layer_call_and_return_conditional_losses_138862"
 dense_50/StatefulPartitionedCall?
 dense_51/StatefulPartitionedCallStatefulPartitionedCall)dense_50/StatefulPartitionedCall:output:0dense_51_14143dense_51_14145*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_51_layer_call_and_return_conditional_losses_139232"
 dense_51/StatefulPartitionedCall?
 dense_52/StatefulPartitionedCallStatefulPartitionedCall)dense_51/StatefulPartitionedCall:output:0dense_52_14148dense_52_14150*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_52_layer_call_and_return_conditional_losses_139602"
 dense_52/StatefulPartitionedCall?
 dense_53/StatefulPartitionedCallStatefulPartitionedCall)dense_52/StatefulPartitionedCall:output:0dense_53_14153dense_53_14155*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_53_layer_call_and_return_conditional_losses_139972"
 dense_53/StatefulPartitionedCall?
flatten_3/PartitionedCallPartitionedCall)dense_53/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_3_layer_call_and_return_conditional_losses_140092
flatten_3/PartitionedCall?
 dense_54/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_54_14159dense_54_14161*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_54_layer_call_and_return_conditional_losses_140222"
 dense_54/StatefulPartitionedCall?
IdentityIdentity)dense_54/StatefulPartitionedCall:output:0!^dense_50/StatefulPartitionedCall!^dense_51/StatefulPartitionedCall!^dense_52/StatefulPartitionedCall!^dense_53/StatefulPartitionedCall!^dense_54/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : 2D
 dense_50/StatefulPartitionedCall dense_50/StatefulPartitionedCall2D
 dense_51/StatefulPartitionedCall dense_51/StatefulPartitionedCall2D
 dense_52/StatefulPartitionedCall dense_52/StatefulPartitionedCall2D
 dense_53/StatefulPartitionedCall dense_53/StatefulPartitionedCall2D
 dense_54/StatefulPartitionedCall dense_54/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?

 __inference__wrapped_model_13848
dense_50_inputK
8sequential_10_dense_50_tensordot_readvariableop_resource:	?E
6sequential_10_dense_50_biasadd_readvariableop_resource:	?L
8sequential_10_dense_51_tensordot_readvariableop_resource:
??E
6sequential_10_dense_51_biasadd_readvariableop_resource:	?L
8sequential_10_dense_52_tensordot_readvariableop_resource:
??E
6sequential_10_dense_52_biasadd_readvariableop_resource:	?K
8sequential_10_dense_53_tensordot_readvariableop_resource:	?@D
6sequential_10_dense_53_biasadd_readvariableop_resource:@I
5sequential_10_dense_54_matmul_readvariableop_resource:
??D
6sequential_10_dense_54_biasadd_readvariableop_resource:
identity??-sequential_10/dense_50/BiasAdd/ReadVariableOp?/sequential_10/dense_50/Tensordot/ReadVariableOp?-sequential_10/dense_51/BiasAdd/ReadVariableOp?/sequential_10/dense_51/Tensordot/ReadVariableOp?-sequential_10/dense_52/BiasAdd/ReadVariableOp?/sequential_10/dense_52/Tensordot/ReadVariableOp?-sequential_10/dense_53/BiasAdd/ReadVariableOp?/sequential_10/dense_53/Tensordot/ReadVariableOp?-sequential_10/dense_54/BiasAdd/ReadVariableOp?,sequential_10/dense_54/MatMul/ReadVariableOp?
/sequential_10/dense_50/Tensordot/ReadVariableOpReadVariableOp8sequential_10_dense_50_tensordot_readvariableop_resource*
_output_shapes
:	?*
dtype021
/sequential_10/dense_50/Tensordot/ReadVariableOp?
%sequential_10/dense_50/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%sequential_10/dense_50/Tensordot/axes?
%sequential_10/dense_50/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%sequential_10/dense_50/Tensordot/free?
&sequential_10/dense_50/Tensordot/ShapeShapedense_50_input*
T0*
_output_shapes
:2(
&sequential_10/dense_50/Tensordot/Shape?
.sequential_10/dense_50/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_10/dense_50/Tensordot/GatherV2/axis?
)sequential_10/dense_50/Tensordot/GatherV2GatherV2/sequential_10/dense_50/Tensordot/Shape:output:0.sequential_10/dense_50/Tensordot/free:output:07sequential_10/dense_50/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_10/dense_50/Tensordot/GatherV2?
0sequential_10/dense_50/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0sequential_10/dense_50/Tensordot/GatherV2_1/axis?
+sequential_10/dense_50/Tensordot/GatherV2_1GatherV2/sequential_10/dense_50/Tensordot/Shape:output:0.sequential_10/dense_50/Tensordot/axes:output:09sequential_10/dense_50/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+sequential_10/dense_50/Tensordot/GatherV2_1?
&sequential_10/dense_50/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_10/dense_50/Tensordot/Const?
%sequential_10/dense_50/Tensordot/ProdProd2sequential_10/dense_50/Tensordot/GatherV2:output:0/sequential_10/dense_50/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%sequential_10/dense_50/Tensordot/Prod?
(sequential_10/dense_50/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_10/dense_50/Tensordot/Const_1?
'sequential_10/dense_50/Tensordot/Prod_1Prod4sequential_10/dense_50/Tensordot/GatherV2_1:output:01sequential_10/dense_50/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'sequential_10/dense_50/Tensordot/Prod_1?
,sequential_10/dense_50/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_10/dense_50/Tensordot/concat/axis?
'sequential_10/dense_50/Tensordot/concatConcatV2.sequential_10/dense_50/Tensordot/free:output:0.sequential_10/dense_50/Tensordot/axes:output:05sequential_10/dense_50/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_10/dense_50/Tensordot/concat?
&sequential_10/dense_50/Tensordot/stackPack.sequential_10/dense_50/Tensordot/Prod:output:00sequential_10/dense_50/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&sequential_10/dense_50/Tensordot/stack?
*sequential_10/dense_50/Tensordot/transpose	Transposedense_50_input0sequential_10/dense_50/Tensordot/concat:output:0*
T0*/
_output_shapes
:?????????2,
*sequential_10/dense_50/Tensordot/transpose?
(sequential_10/dense_50/Tensordot/ReshapeReshape.sequential_10/dense_50/Tensordot/transpose:y:0/sequential_10/dense_50/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2*
(sequential_10/dense_50/Tensordot/Reshape?
'sequential_10/dense_50/Tensordot/MatMulMatMul1sequential_10/dense_50/Tensordot/Reshape:output:07sequential_10/dense_50/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2)
'sequential_10/dense_50/Tensordot/MatMul?
(sequential_10/dense_50/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2*
(sequential_10/dense_50/Tensordot/Const_2?
.sequential_10/dense_50/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_10/dense_50/Tensordot/concat_1/axis?
)sequential_10/dense_50/Tensordot/concat_1ConcatV22sequential_10/dense_50/Tensordot/GatherV2:output:01sequential_10/dense_50/Tensordot/Const_2:output:07sequential_10/dense_50/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)sequential_10/dense_50/Tensordot/concat_1?
 sequential_10/dense_50/TensordotReshape1sequential_10/dense_50/Tensordot/MatMul:product:02sequential_10/dense_50/Tensordot/concat_1:output:0*
T0*0
_output_shapes
:??????????2"
 sequential_10/dense_50/Tensordot?
-sequential_10/dense_50/BiasAdd/ReadVariableOpReadVariableOp6sequential_10_dense_50_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-sequential_10/dense_50/BiasAdd/ReadVariableOp?
sequential_10/dense_50/BiasAddBiasAdd)sequential_10/dense_50/Tensordot:output:05sequential_10/dense_50/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2 
sequential_10/dense_50/BiasAdd?
sequential_10/dense_50/ReluRelu'sequential_10/dense_50/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
sequential_10/dense_50/Relu?
/sequential_10/dense_51/Tensordot/ReadVariableOpReadVariableOp8sequential_10_dense_51_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype021
/sequential_10/dense_51/Tensordot/ReadVariableOp?
%sequential_10/dense_51/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%sequential_10/dense_51/Tensordot/axes?
%sequential_10/dense_51/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%sequential_10/dense_51/Tensordot/free?
&sequential_10/dense_51/Tensordot/ShapeShape)sequential_10/dense_50/Relu:activations:0*
T0*
_output_shapes
:2(
&sequential_10/dense_51/Tensordot/Shape?
.sequential_10/dense_51/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_10/dense_51/Tensordot/GatherV2/axis?
)sequential_10/dense_51/Tensordot/GatherV2GatherV2/sequential_10/dense_51/Tensordot/Shape:output:0.sequential_10/dense_51/Tensordot/free:output:07sequential_10/dense_51/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_10/dense_51/Tensordot/GatherV2?
0sequential_10/dense_51/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0sequential_10/dense_51/Tensordot/GatherV2_1/axis?
+sequential_10/dense_51/Tensordot/GatherV2_1GatherV2/sequential_10/dense_51/Tensordot/Shape:output:0.sequential_10/dense_51/Tensordot/axes:output:09sequential_10/dense_51/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+sequential_10/dense_51/Tensordot/GatherV2_1?
&sequential_10/dense_51/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_10/dense_51/Tensordot/Const?
%sequential_10/dense_51/Tensordot/ProdProd2sequential_10/dense_51/Tensordot/GatherV2:output:0/sequential_10/dense_51/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%sequential_10/dense_51/Tensordot/Prod?
(sequential_10/dense_51/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_10/dense_51/Tensordot/Const_1?
'sequential_10/dense_51/Tensordot/Prod_1Prod4sequential_10/dense_51/Tensordot/GatherV2_1:output:01sequential_10/dense_51/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'sequential_10/dense_51/Tensordot/Prod_1?
,sequential_10/dense_51/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_10/dense_51/Tensordot/concat/axis?
'sequential_10/dense_51/Tensordot/concatConcatV2.sequential_10/dense_51/Tensordot/free:output:0.sequential_10/dense_51/Tensordot/axes:output:05sequential_10/dense_51/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_10/dense_51/Tensordot/concat?
&sequential_10/dense_51/Tensordot/stackPack.sequential_10/dense_51/Tensordot/Prod:output:00sequential_10/dense_51/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&sequential_10/dense_51/Tensordot/stack?
*sequential_10/dense_51/Tensordot/transpose	Transpose)sequential_10/dense_50/Relu:activations:00sequential_10/dense_51/Tensordot/concat:output:0*
T0*0
_output_shapes
:??????????2,
*sequential_10/dense_51/Tensordot/transpose?
(sequential_10/dense_51/Tensordot/ReshapeReshape.sequential_10/dense_51/Tensordot/transpose:y:0/sequential_10/dense_51/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2*
(sequential_10/dense_51/Tensordot/Reshape?
'sequential_10/dense_51/Tensordot/MatMulMatMul1sequential_10/dense_51/Tensordot/Reshape:output:07sequential_10/dense_51/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2)
'sequential_10/dense_51/Tensordot/MatMul?
(sequential_10/dense_51/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2*
(sequential_10/dense_51/Tensordot/Const_2?
.sequential_10/dense_51/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_10/dense_51/Tensordot/concat_1/axis?
)sequential_10/dense_51/Tensordot/concat_1ConcatV22sequential_10/dense_51/Tensordot/GatherV2:output:01sequential_10/dense_51/Tensordot/Const_2:output:07sequential_10/dense_51/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)sequential_10/dense_51/Tensordot/concat_1?
 sequential_10/dense_51/TensordotReshape1sequential_10/dense_51/Tensordot/MatMul:product:02sequential_10/dense_51/Tensordot/concat_1:output:0*
T0*0
_output_shapes
:??????????2"
 sequential_10/dense_51/Tensordot?
-sequential_10/dense_51/BiasAdd/ReadVariableOpReadVariableOp6sequential_10_dense_51_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-sequential_10/dense_51/BiasAdd/ReadVariableOp?
sequential_10/dense_51/BiasAddBiasAdd)sequential_10/dense_51/Tensordot:output:05sequential_10/dense_51/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2 
sequential_10/dense_51/BiasAdd?
sequential_10/dense_51/ReluRelu'sequential_10/dense_51/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
sequential_10/dense_51/Relu?
/sequential_10/dense_52/Tensordot/ReadVariableOpReadVariableOp8sequential_10_dense_52_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype021
/sequential_10/dense_52/Tensordot/ReadVariableOp?
%sequential_10/dense_52/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%sequential_10/dense_52/Tensordot/axes?
%sequential_10/dense_52/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%sequential_10/dense_52/Tensordot/free?
&sequential_10/dense_52/Tensordot/ShapeShape)sequential_10/dense_51/Relu:activations:0*
T0*
_output_shapes
:2(
&sequential_10/dense_52/Tensordot/Shape?
.sequential_10/dense_52/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_10/dense_52/Tensordot/GatherV2/axis?
)sequential_10/dense_52/Tensordot/GatherV2GatherV2/sequential_10/dense_52/Tensordot/Shape:output:0.sequential_10/dense_52/Tensordot/free:output:07sequential_10/dense_52/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_10/dense_52/Tensordot/GatherV2?
0sequential_10/dense_52/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0sequential_10/dense_52/Tensordot/GatherV2_1/axis?
+sequential_10/dense_52/Tensordot/GatherV2_1GatherV2/sequential_10/dense_52/Tensordot/Shape:output:0.sequential_10/dense_52/Tensordot/axes:output:09sequential_10/dense_52/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+sequential_10/dense_52/Tensordot/GatherV2_1?
&sequential_10/dense_52/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_10/dense_52/Tensordot/Const?
%sequential_10/dense_52/Tensordot/ProdProd2sequential_10/dense_52/Tensordot/GatherV2:output:0/sequential_10/dense_52/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%sequential_10/dense_52/Tensordot/Prod?
(sequential_10/dense_52/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_10/dense_52/Tensordot/Const_1?
'sequential_10/dense_52/Tensordot/Prod_1Prod4sequential_10/dense_52/Tensordot/GatherV2_1:output:01sequential_10/dense_52/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'sequential_10/dense_52/Tensordot/Prod_1?
,sequential_10/dense_52/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_10/dense_52/Tensordot/concat/axis?
'sequential_10/dense_52/Tensordot/concatConcatV2.sequential_10/dense_52/Tensordot/free:output:0.sequential_10/dense_52/Tensordot/axes:output:05sequential_10/dense_52/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_10/dense_52/Tensordot/concat?
&sequential_10/dense_52/Tensordot/stackPack.sequential_10/dense_52/Tensordot/Prod:output:00sequential_10/dense_52/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&sequential_10/dense_52/Tensordot/stack?
*sequential_10/dense_52/Tensordot/transpose	Transpose)sequential_10/dense_51/Relu:activations:00sequential_10/dense_52/Tensordot/concat:output:0*
T0*0
_output_shapes
:??????????2,
*sequential_10/dense_52/Tensordot/transpose?
(sequential_10/dense_52/Tensordot/ReshapeReshape.sequential_10/dense_52/Tensordot/transpose:y:0/sequential_10/dense_52/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2*
(sequential_10/dense_52/Tensordot/Reshape?
'sequential_10/dense_52/Tensordot/MatMulMatMul1sequential_10/dense_52/Tensordot/Reshape:output:07sequential_10/dense_52/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2)
'sequential_10/dense_52/Tensordot/MatMul?
(sequential_10/dense_52/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2*
(sequential_10/dense_52/Tensordot/Const_2?
.sequential_10/dense_52/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_10/dense_52/Tensordot/concat_1/axis?
)sequential_10/dense_52/Tensordot/concat_1ConcatV22sequential_10/dense_52/Tensordot/GatherV2:output:01sequential_10/dense_52/Tensordot/Const_2:output:07sequential_10/dense_52/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)sequential_10/dense_52/Tensordot/concat_1?
 sequential_10/dense_52/TensordotReshape1sequential_10/dense_52/Tensordot/MatMul:product:02sequential_10/dense_52/Tensordot/concat_1:output:0*
T0*0
_output_shapes
:??????????2"
 sequential_10/dense_52/Tensordot?
-sequential_10/dense_52/BiasAdd/ReadVariableOpReadVariableOp6sequential_10_dense_52_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-sequential_10/dense_52/BiasAdd/ReadVariableOp?
sequential_10/dense_52/BiasAddBiasAdd)sequential_10/dense_52/Tensordot:output:05sequential_10/dense_52/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2 
sequential_10/dense_52/BiasAdd?
sequential_10/dense_52/ReluRelu'sequential_10/dense_52/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
sequential_10/dense_52/Relu?
/sequential_10/dense_53/Tensordot/ReadVariableOpReadVariableOp8sequential_10_dense_53_tensordot_readvariableop_resource*
_output_shapes
:	?@*
dtype021
/sequential_10/dense_53/Tensordot/ReadVariableOp?
%sequential_10/dense_53/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%sequential_10/dense_53/Tensordot/axes?
%sequential_10/dense_53/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%sequential_10/dense_53/Tensordot/free?
&sequential_10/dense_53/Tensordot/ShapeShape)sequential_10/dense_52/Relu:activations:0*
T0*
_output_shapes
:2(
&sequential_10/dense_53/Tensordot/Shape?
.sequential_10/dense_53/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_10/dense_53/Tensordot/GatherV2/axis?
)sequential_10/dense_53/Tensordot/GatherV2GatherV2/sequential_10/dense_53/Tensordot/Shape:output:0.sequential_10/dense_53/Tensordot/free:output:07sequential_10/dense_53/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_10/dense_53/Tensordot/GatherV2?
0sequential_10/dense_53/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0sequential_10/dense_53/Tensordot/GatherV2_1/axis?
+sequential_10/dense_53/Tensordot/GatherV2_1GatherV2/sequential_10/dense_53/Tensordot/Shape:output:0.sequential_10/dense_53/Tensordot/axes:output:09sequential_10/dense_53/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+sequential_10/dense_53/Tensordot/GatherV2_1?
&sequential_10/dense_53/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_10/dense_53/Tensordot/Const?
%sequential_10/dense_53/Tensordot/ProdProd2sequential_10/dense_53/Tensordot/GatherV2:output:0/sequential_10/dense_53/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%sequential_10/dense_53/Tensordot/Prod?
(sequential_10/dense_53/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_10/dense_53/Tensordot/Const_1?
'sequential_10/dense_53/Tensordot/Prod_1Prod4sequential_10/dense_53/Tensordot/GatherV2_1:output:01sequential_10/dense_53/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'sequential_10/dense_53/Tensordot/Prod_1?
,sequential_10/dense_53/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_10/dense_53/Tensordot/concat/axis?
'sequential_10/dense_53/Tensordot/concatConcatV2.sequential_10/dense_53/Tensordot/free:output:0.sequential_10/dense_53/Tensordot/axes:output:05sequential_10/dense_53/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_10/dense_53/Tensordot/concat?
&sequential_10/dense_53/Tensordot/stackPack.sequential_10/dense_53/Tensordot/Prod:output:00sequential_10/dense_53/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&sequential_10/dense_53/Tensordot/stack?
*sequential_10/dense_53/Tensordot/transpose	Transpose)sequential_10/dense_52/Relu:activations:00sequential_10/dense_53/Tensordot/concat:output:0*
T0*0
_output_shapes
:??????????2,
*sequential_10/dense_53/Tensordot/transpose?
(sequential_10/dense_53/Tensordot/ReshapeReshape.sequential_10/dense_53/Tensordot/transpose:y:0/sequential_10/dense_53/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2*
(sequential_10/dense_53/Tensordot/Reshape?
'sequential_10/dense_53/Tensordot/MatMulMatMul1sequential_10/dense_53/Tensordot/Reshape:output:07sequential_10/dense_53/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2)
'sequential_10/dense_53/Tensordot/MatMul?
(sequential_10/dense_53/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2*
(sequential_10/dense_53/Tensordot/Const_2?
.sequential_10/dense_53/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_10/dense_53/Tensordot/concat_1/axis?
)sequential_10/dense_53/Tensordot/concat_1ConcatV22sequential_10/dense_53/Tensordot/GatherV2:output:01sequential_10/dense_53/Tensordot/Const_2:output:07sequential_10/dense_53/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)sequential_10/dense_53/Tensordot/concat_1?
 sequential_10/dense_53/TensordotReshape1sequential_10/dense_53/Tensordot/MatMul:product:02sequential_10/dense_53/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:?????????@2"
 sequential_10/dense_53/Tensordot?
-sequential_10/dense_53/BiasAdd/ReadVariableOpReadVariableOp6sequential_10_dense_53_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-sequential_10/dense_53/BiasAdd/ReadVariableOp?
sequential_10/dense_53/BiasAddBiasAdd)sequential_10/dense_53/Tensordot:output:05sequential_10/dense_53/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2 
sequential_10/dense_53/BiasAdd?
sequential_10/dense_53/ReluRelu'sequential_10/dense_53/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
sequential_10/dense_53/Relu?
sequential_10/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"????@n  2
sequential_10/flatten_3/Const?
sequential_10/flatten_3/ReshapeReshape)sequential_10/dense_53/Relu:activations:0&sequential_10/flatten_3/Const:output:0*
T0*)
_output_shapes
:???????????2!
sequential_10/flatten_3/Reshape?
,sequential_10/dense_54/MatMul/ReadVariableOpReadVariableOp5sequential_10_dense_54_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02.
,sequential_10/dense_54/MatMul/ReadVariableOp?
sequential_10/dense_54/MatMulMatMul(sequential_10/flatten_3/Reshape:output:04sequential_10/dense_54/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_10/dense_54/MatMul?
-sequential_10/dense_54/BiasAdd/ReadVariableOpReadVariableOp6sequential_10_dense_54_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_10/dense_54/BiasAdd/ReadVariableOp?
sequential_10/dense_54/BiasAddBiasAdd'sequential_10/dense_54/MatMul:product:05sequential_10/dense_54/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2 
sequential_10/dense_54/BiasAdd?
sequential_10/dense_54/SoftmaxSoftmax'sequential_10/dense_54/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2 
sequential_10/dense_54/Softmax?
IdentityIdentity(sequential_10/dense_54/Softmax:softmax:0.^sequential_10/dense_50/BiasAdd/ReadVariableOp0^sequential_10/dense_50/Tensordot/ReadVariableOp.^sequential_10/dense_51/BiasAdd/ReadVariableOp0^sequential_10/dense_51/Tensordot/ReadVariableOp.^sequential_10/dense_52/BiasAdd/ReadVariableOp0^sequential_10/dense_52/Tensordot/ReadVariableOp.^sequential_10/dense_53/BiasAdd/ReadVariableOp0^sequential_10/dense_53/Tensordot/ReadVariableOp.^sequential_10/dense_54/BiasAdd/ReadVariableOp-^sequential_10/dense_54/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : 2^
-sequential_10/dense_50/BiasAdd/ReadVariableOp-sequential_10/dense_50/BiasAdd/ReadVariableOp2b
/sequential_10/dense_50/Tensordot/ReadVariableOp/sequential_10/dense_50/Tensordot/ReadVariableOp2^
-sequential_10/dense_51/BiasAdd/ReadVariableOp-sequential_10/dense_51/BiasAdd/ReadVariableOp2b
/sequential_10/dense_51/Tensordot/ReadVariableOp/sequential_10/dense_51/Tensordot/ReadVariableOp2^
-sequential_10/dense_52/BiasAdd/ReadVariableOp-sequential_10/dense_52/BiasAdd/ReadVariableOp2b
/sequential_10/dense_52/Tensordot/ReadVariableOp/sequential_10/dense_52/Tensordot/ReadVariableOp2^
-sequential_10/dense_53/BiasAdd/ReadVariableOp-sequential_10/dense_53/BiasAdd/ReadVariableOp2b
/sequential_10/dense_53/Tensordot/ReadVariableOp/sequential_10/dense_53/Tensordot/ReadVariableOp2^
-sequential_10/dense_54/BiasAdd/ReadVariableOp-sequential_10/dense_54/BiasAdd/ReadVariableOp2\
,sequential_10/dense_54/MatMul/ReadVariableOp,sequential_10/dense_54/MatMul/ReadVariableOp:_ [
/
_output_shapes
:?????????
(
_user_specified_namedense_50_input
?
?
H__inference_sequential_10_layer_call_and_return_conditional_losses_14243
dense_50_input!
dense_50_14216:	?
dense_50_14218:	?"
dense_51_14221:
??
dense_51_14223:	?"
dense_52_14226:
??
dense_52_14228:	?!
dense_53_14231:	?@
dense_53_14233:@"
dense_54_14237:
??
dense_54_14239:
identity?? dense_50/StatefulPartitionedCall? dense_51/StatefulPartitionedCall? dense_52/StatefulPartitionedCall? dense_53/StatefulPartitionedCall? dense_54/StatefulPartitionedCall?
 dense_50/StatefulPartitionedCallStatefulPartitionedCalldense_50_inputdense_50_14216dense_50_14218*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_50_layer_call_and_return_conditional_losses_138862"
 dense_50/StatefulPartitionedCall?
 dense_51/StatefulPartitionedCallStatefulPartitionedCall)dense_50/StatefulPartitionedCall:output:0dense_51_14221dense_51_14223*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_51_layer_call_and_return_conditional_losses_139232"
 dense_51/StatefulPartitionedCall?
 dense_52/StatefulPartitionedCallStatefulPartitionedCall)dense_51/StatefulPartitionedCall:output:0dense_52_14226dense_52_14228*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_52_layer_call_and_return_conditional_losses_139602"
 dense_52/StatefulPartitionedCall?
 dense_53/StatefulPartitionedCallStatefulPartitionedCall)dense_52/StatefulPartitionedCall:output:0dense_53_14231dense_53_14233*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_53_layer_call_and_return_conditional_losses_139972"
 dense_53/StatefulPartitionedCall?
flatten_3/PartitionedCallPartitionedCall)dense_53/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_3_layer_call_and_return_conditional_losses_140092
flatten_3/PartitionedCall?
 dense_54/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_54_14237dense_54_14239*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_54_layer_call_and_return_conditional_losses_140222"
 dense_54/StatefulPartitionedCall?
IdentityIdentity)dense_54/StatefulPartitionedCall:output:0!^dense_50/StatefulPartitionedCall!^dense_51/StatefulPartitionedCall!^dense_52/StatefulPartitionedCall!^dense_53/StatefulPartitionedCall!^dense_54/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : 2D
 dense_50/StatefulPartitionedCall dense_50/StatefulPartitionedCall2D
 dense_51/StatefulPartitionedCall dense_51/StatefulPartitionedCall2D
 dense_52/StatefulPartitionedCall dense_52/StatefulPartitionedCall2D
 dense_53/StatefulPartitionedCall dense_53/StatefulPartitionedCall2D
 dense_54/StatefulPartitionedCall dense_54/StatefulPartitionedCall:_ [
/
_output_shapes
:?????????
(
_user_specified_namedense_50_input
?!
?
C__inference_dense_51_layer_call_and_return_conditional_losses_13923

inputs5
!tensordot_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesu
Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*0
_output_shapes
:??????????2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*0
_output_shapes
:??????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
(__inference_dense_50_layer_call_fn_14638

inputs
unknown:	?
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_50_layer_call_and_return_conditional_losses_138862
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
(__inference_dense_53_layer_call_fn_14758

inputs
unknown:	?@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_53_layer_call_and_return_conditional_losses_139972
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
Q
dense_50_input?
 serving_default_dense_50_input:0?????????<
dense_540
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?8
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer-4
layer_with_weights-4
layer-5
	optimizer
	variables
	regularization_losses

trainable_variables
	keras_api

signatures
*l&call_and_return_all_conditional_losses
m__call__
n_default_save_signature"?5
_tf_keras_sequential?5{"name": "sequential_10", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_10", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 21, 21, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_50_input"}}, {"class_name": "Dense", "config": {"name": "dense_50", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 21, 21, 3]}, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_51", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_52", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_53", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Flatten", "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_54", "trainable": true, "dtype": "float32", "units": 27, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "shared_object_id": 17, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 3}}, "shared_object_id": 18}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 21, 21, 3]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 21, 21, 3]}, "float32", "dense_50_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_10", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 21, 21, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_50_input"}, "shared_object_id": 0}, {"class_name": "Dense", "config": {"name": "dense_50", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 21, 21, 3]}, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3}, {"class_name": "Dense", "config": {"name": "dense_51", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 6}, {"class_name": "Dense", "config": {"name": "dense_52", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 9}, {"class_name": "Dense", "config": {"name": "dense_53", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 11}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 12}, {"class_name": "Flatten", "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 13}, {"class_name": "Dense", "config": {"name": "dense_54", "trainable": true, "dtype": "float32", "units": 27, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 14}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 15}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 16}]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 19}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "RMSprop", "config": {"name": "RMSprop", "learning_rate": 0.0010000000474974513, "decay": 0.0, "rho": 0.8999999761581421, "momentum": 0.0, "epsilon": 1e-07, "centered": false}}}}
?	

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
*o&call_and_return_all_conditional_losses
p__call__"?
_tf_keras_layer?{"name": "dense_50", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 21, 21, 3]}, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_50", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 21, 21, 3]}, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 3}}, "shared_object_id": 18}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 21, 21, 3]}}
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
*q&call_and_return_all_conditional_losses
r__call__"?
_tf_keras_layer?{"name": "dense_51", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_51", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}, "shared_object_id": 20}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 21, 21, 512]}}
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
*s&call_and_return_all_conditional_losses
t__call__"?
_tf_keras_layer?{"name": "dense_52", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_52", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}, "shared_object_id": 21}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 21, 21, 256]}}
?

kernel
 bias
!	variables
"trainable_variables
#regularization_losses
$	keras_api
*u&call_and_return_all_conditional_losses
v__call__"?
_tf_keras_layer?{"name": "dense_53", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_53", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 11}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 12, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}, "shared_object_id": 22}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 21, 21, 128]}}
?
%	variables
&trainable_variables
'regularization_losses
(	keras_api
*w&call_and_return_all_conditional_losses
x__call__"?
_tf_keras_layer?{"name": "flatten_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 13, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 23}}
?

)kernel
*bias
+	variables
,trainable_variables
-regularization_losses
.	keras_api
*y&call_and_return_all_conditional_losses
z__call__"?
_tf_keras_layer?{"name": "dense_54", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_54", "trainable": true, "dtype": "float32", "units": 27, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 14}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 15}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 16, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 28224}}, "shared_object_id": 24}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28224]}}
?
/iter
	0decay
1learning_rate
2momentum
3rho	rmsb	rmsc	rmsd	rmse	rmsf	rmsg	rmsh	 rmsi	)rmsj	*rmsk"
	optimizer
f
0
1
2
3
4
5
6
 7
)8
*9"
trackable_list_wrapper
 "
trackable_list_wrapper
f
0
1
2
3
4
5
6
 7
)8
*9"
trackable_list_wrapper
?

4layers
	variables
5layer_metrics
	regularization_losses

trainable_variables
6layer_regularization_losses
7metrics
8non_trainable_variables
m__call__
n_default_save_signature
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
,
{serving_default"
signature_map
": 	?2dense_50/kernel
:?2dense_50/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?

9layers
	variables
:layer_metrics
trainable_variables
regularization_losses
;layer_regularization_losses
<metrics
=non_trainable_variables
p__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses"
_generic_user_object
#:!
??2dense_51/kernel
:?2dense_51/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?

>layers
	variables
?layer_metrics
trainable_variables
regularization_losses
@layer_regularization_losses
Ametrics
Bnon_trainable_variables
r__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses"
_generic_user_object
#:!
??2dense_52/kernel
:?2dense_52/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?

Clayers
	variables
Dlayer_metrics
trainable_variables
regularization_losses
Elayer_regularization_losses
Fmetrics
Gnon_trainable_variables
t__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
": 	?@2dense_53/kernel
:@2dense_53/bias
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
?

Hlayers
!	variables
Ilayer_metrics
"trainable_variables
#regularization_losses
Jlayer_regularization_losses
Kmetrics
Lnon_trainable_variables
v__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

Mlayers
%	variables
Nlayer_metrics
&trainable_variables
'regularization_losses
Olayer_regularization_losses
Pmetrics
Qnon_trainable_variables
x__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
#:!
??2dense_54/kernel
:2dense_54/bias
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
?

Rlayers
+	variables
Slayer_metrics
,trainable_variables
-regularization_losses
Tlayer_regularization_losses
Umetrics
Vnon_trainable_variables
z__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
:	 (2RMSprop/iter
: (2RMSprop/decay
: (2RMSprop/learning_rate
: (2RMSprop/momentum
: (2RMSprop/rho
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
W0
X1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	Ytotal
	Zcount
[	variables
\	keras_api"?
_tf_keras_metric?{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 25}
?
	]total
	^count
_
_fn_kwargs
`	variables
a	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 19}
:  (2total
:  (2count
.
Y0
Z1"
trackable_list_wrapper
-
[	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
]0
^1"
trackable_list_wrapper
-
`	variables"
_generic_user_object
,:*	?2RMSprop/dense_50/kernel/rms
&:$?2RMSprop/dense_50/bias/rms
-:+
??2RMSprop/dense_51/kernel/rms
&:$?2RMSprop/dense_51/bias/rms
-:+
??2RMSprop/dense_52/kernel/rms
&:$?2RMSprop/dense_52/bias/rms
,:*	?@2RMSprop/dense_53/kernel/rms
%:#@2RMSprop/dense_53/bias/rms
-:+
??2RMSprop/dense_54/kernel/rms
%:#2RMSprop/dense_54/bias/rms
?2?
H__inference_sequential_10_layer_call_and_return_conditional_losses_14427
H__inference_sequential_10_layer_call_and_return_conditional_losses_14548
H__inference_sequential_10_layer_call_and_return_conditional_losses_14243
H__inference_sequential_10_layer_call_and_return_conditional_losses_14273?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
-__inference_sequential_10_layer_call_fn_14052
-__inference_sequential_10_layer_call_fn_14573
-__inference_sequential_10_layer_call_fn_14598
-__inference_sequential_10_layer_call_fn_14213?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
 __inference__wrapped_model_13848?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *5?2
0?-
dense_50_input?????????
?2?
C__inference_dense_50_layer_call_and_return_conditional_losses_14629?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_50_layer_call_fn_14638?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_51_layer_call_and_return_conditional_losses_14669?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_51_layer_call_fn_14678?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_52_layer_call_and_return_conditional_losses_14709?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_52_layer_call_fn_14718?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_53_layer_call_and_return_conditional_losses_14749?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_53_layer_call_fn_14758?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_flatten_3_layer_call_and_return_conditional_losses_14764?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_flatten_3_layer_call_fn_14769?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_54_layer_call_and_return_conditional_losses_14780?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_54_layer_call_fn_14789?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference_signature_wrapper_14306dense_50_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
 __inference__wrapped_model_13848?
 )*??<
5?2
0?-
dense_50_input?????????
? "3?0
.
dense_54"?
dense_54??????????
C__inference_dense_50_layer_call_and_return_conditional_losses_14629m7?4
-?*
(?%
inputs?????????
? ".?+
$?!
0??????????
? ?
(__inference_dense_50_layer_call_fn_14638`7?4
-?*
(?%
inputs?????????
? "!????????????
C__inference_dense_51_layer_call_and_return_conditional_losses_14669n8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
(__inference_dense_51_layer_call_fn_14678a8?5
.?+
)?&
inputs??????????
? "!????????????
C__inference_dense_52_layer_call_and_return_conditional_losses_14709n8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
(__inference_dense_52_layer_call_fn_14718a8?5
.?+
)?&
inputs??????????
? "!????????????
C__inference_dense_53_layer_call_and_return_conditional_losses_14749m 8?5
.?+
)?&
inputs??????????
? "-?*
#? 
0?????????@
? ?
(__inference_dense_53_layer_call_fn_14758` 8?5
.?+
)?&
inputs??????????
? " ??????????@?
C__inference_dense_54_layer_call_and_return_conditional_losses_14780^)*1?.
'?$
"?
inputs???????????
? "%?"
?
0?????????
? }
(__inference_dense_54_layer_call_fn_14789Q)*1?.
'?$
"?
inputs???????????
? "???????????
D__inference_flatten_3_layer_call_and_return_conditional_losses_14764b7?4
-?*
(?%
inputs?????????@
? "'?$
?
0???????????
? ?
)__inference_flatten_3_layer_call_fn_14769U7?4
-?*
(?%
inputs?????????@
? "?????????????
H__inference_sequential_10_layer_call_and_return_conditional_losses_14243|
 )*G?D
=?:
0?-
dense_50_input?????????
p 

 
? "%?"
?
0?????????
? ?
H__inference_sequential_10_layer_call_and_return_conditional_losses_14273|
 )*G?D
=?:
0?-
dense_50_input?????????
p

 
? "%?"
?
0?????????
? ?
H__inference_sequential_10_layer_call_and_return_conditional_losses_14427t
 )*??<
5?2
(?%
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
H__inference_sequential_10_layer_call_and_return_conditional_losses_14548t
 )*??<
5?2
(?%
inputs?????????
p

 
? "%?"
?
0?????????
? ?
-__inference_sequential_10_layer_call_fn_14052o
 )*G?D
=?:
0?-
dense_50_input?????????
p 

 
? "???????????
-__inference_sequential_10_layer_call_fn_14213o
 )*G?D
=?:
0?-
dense_50_input?????????
p

 
? "???????????
-__inference_sequential_10_layer_call_fn_14573g
 )*??<
5?2
(?%
inputs?????????
p 

 
? "???????????
-__inference_sequential_10_layer_call_fn_14598g
 )*??<
5?2
(?%
inputs?????????
p

 
? "???????????
#__inference_signature_wrapper_14306?
 )*Q?N
? 
G?D
B
dense_50_input0?-
dense_50_input?????????"3?0
.
dense_54"?
dense_54?????????