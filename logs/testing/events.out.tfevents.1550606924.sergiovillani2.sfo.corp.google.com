       �K"	   ��Abrain.Event:2�~gM�      ��	8]/��A"��
t
input/PlaceholderPlaceholder*
dtype0*'
_output_shapes
:���������*
shape:���������
�
1layer_1/weights1/Initializer/random_uniform/shapeConst*#
_class
loc:@layer_1/weights1*
valueB"      *
dtype0*
_output_shapes
:
�
/layer_1/weights1/Initializer/random_uniform/minConst*#
_class
loc:@layer_1/weights1*
valueB
 *�7��*
dtype0*
_output_shapes
: 
�
/layer_1/weights1/Initializer/random_uniform/maxConst*#
_class
loc:@layer_1/weights1*
valueB
 *�7�?*
dtype0*
_output_shapes
: 
�
9layer_1/weights1/Initializer/random_uniform/RandomUniformRandomUniform1layer_1/weights1/Initializer/random_uniform/shape*
T0*#
_class
loc:@layer_1/weights1*
seed2 *
dtype0*
_output_shapes

:*

seed 
�
/layer_1/weights1/Initializer/random_uniform/subSub/layer_1/weights1/Initializer/random_uniform/max/layer_1/weights1/Initializer/random_uniform/min*
_output_shapes
: *
T0*#
_class
loc:@layer_1/weights1
�
/layer_1/weights1/Initializer/random_uniform/mulMul9layer_1/weights1/Initializer/random_uniform/RandomUniform/layer_1/weights1/Initializer/random_uniform/sub*
T0*#
_class
loc:@layer_1/weights1*
_output_shapes

:
�
+layer_1/weights1/Initializer/random_uniformAdd/layer_1/weights1/Initializer/random_uniform/mul/layer_1/weights1/Initializer/random_uniform/min*
T0*#
_class
loc:@layer_1/weights1*
_output_shapes

:
�
layer_1/weights1
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@layer_1/weights1*
	container 
�
layer_1/weights1/AssignAssignlayer_1/weights1+layer_1/weights1/Initializer/random_uniform*
use_locking(*
T0*#
_class
loc:@layer_1/weights1*
validate_shape(*
_output_shapes

:
�
layer_1/weights1/readIdentitylayer_1/weights1*
T0*#
_class
loc:@layer_1/weights1*
_output_shapes

:
�
!layer_1/biases1/Initializer/zerosConst*"
_class
loc:@layer_1/biases1*
valueB*    *
dtype0*
_output_shapes
:
�
layer_1/biases1
VariableV2*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@layer_1/biases1*
	container *
shape:
�
layer_1/biases1/AssignAssignlayer_1/biases1!layer_1/biases1/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@layer_1/biases1
z
layer_1/biases1/readIdentitylayer_1/biases1*
T0*"
_class
loc:@layer_1/biases1*
_output_shapes
:
�
layer_1/MatMulMatMulinput/Placeholderlayer_1/weights1/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
j
layer_1/addAddlayer_1/MatMullayer_1/biases1/read*
T0*'
_output_shapes
:���������
S
layer_1/ReluRelulayer_1/add*
T0*'
_output_shapes
:���������
�
1layer_2/weights2/Initializer/random_uniform/shapeConst*#
_class
loc:@layer_2/weights2*
valueB"      *
dtype0*
_output_shapes
:
�
/layer_2/weights2/Initializer/random_uniform/minConst*#
_class
loc:@layer_2/weights2*
valueB
 *׳]�*
dtype0*
_output_shapes
: 
�
/layer_2/weights2/Initializer/random_uniform/maxConst*#
_class
loc:@layer_2/weights2*
valueB
 *׳]?*
dtype0*
_output_shapes
: 
�
9layer_2/weights2/Initializer/random_uniform/RandomUniformRandomUniform1layer_2/weights2/Initializer/random_uniform/shape*

seed *
T0*#
_class
loc:@layer_2/weights2*
seed2 *
dtype0*
_output_shapes

:
�
/layer_2/weights2/Initializer/random_uniform/subSub/layer_2/weights2/Initializer/random_uniform/max/layer_2/weights2/Initializer/random_uniform/min*
_output_shapes
: *
T0*#
_class
loc:@layer_2/weights2
�
/layer_2/weights2/Initializer/random_uniform/mulMul9layer_2/weights2/Initializer/random_uniform/RandomUniform/layer_2/weights2/Initializer/random_uniform/sub*
T0*#
_class
loc:@layer_2/weights2*
_output_shapes

:
�
+layer_2/weights2/Initializer/random_uniformAdd/layer_2/weights2/Initializer/random_uniform/mul/layer_2/weights2/Initializer/random_uniform/min*
_output_shapes

:*
T0*#
_class
loc:@layer_2/weights2
�
layer_2/weights2
VariableV2*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@layer_2/weights2*
	container *
shape
:
�
layer_2/weights2/AssignAssignlayer_2/weights2+layer_2/weights2/Initializer/random_uniform*
use_locking(*
T0*#
_class
loc:@layer_2/weights2*
validate_shape(*
_output_shapes

:
�
layer_2/weights2/readIdentitylayer_2/weights2*
_output_shapes

:*
T0*#
_class
loc:@layer_2/weights2
�
!layer_2/biases2/Initializer/zerosConst*
dtype0*
_output_shapes
:*"
_class
loc:@layer_2/biases2*
valueB*    
�
layer_2/biases2
VariableV2*"
_class
loc:@layer_2/biases2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
�
layer_2/biases2/AssignAssignlayer_2/biases2!layer_2/biases2/Initializer/zeros*
T0*"
_class
loc:@layer_2/biases2*
validate_shape(*
_output_shapes
:*
use_locking(
z
layer_2/biases2/readIdentitylayer_2/biases2*
T0*"
_class
loc:@layer_2/biases2*
_output_shapes
:
�
layer_2/MatMulMatMullayer_1/Relulayer_2/weights2/read*
transpose_b( *
T0*'
_output_shapes
:���������*
transpose_a( 
j
layer_2/addAddlayer_2/MatMullayer_2/biases2/read*
T0*'
_output_shapes
:���������
S
layer_2/ReluRelulayer_2/add*
T0*'
_output_shapes
:���������
�
1layer_3/weights3/Initializer/random_uniform/shapeConst*#
_class
loc:@layer_3/weights3*
valueB"      *
dtype0*
_output_shapes
:
�
/layer_3/weights3/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *#
_class
loc:@layer_3/weights3*
valueB
 *׳]�
�
/layer_3/weights3/Initializer/random_uniform/maxConst*#
_class
loc:@layer_3/weights3*
valueB
 *׳]?*
dtype0*
_output_shapes
: 
�
9layer_3/weights3/Initializer/random_uniform/RandomUniformRandomUniform1layer_3/weights3/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:*

seed *
T0*#
_class
loc:@layer_3/weights3*
seed2 
�
/layer_3/weights3/Initializer/random_uniform/subSub/layer_3/weights3/Initializer/random_uniform/max/layer_3/weights3/Initializer/random_uniform/min*
_output_shapes
: *
T0*#
_class
loc:@layer_3/weights3
�
/layer_3/weights3/Initializer/random_uniform/mulMul9layer_3/weights3/Initializer/random_uniform/RandomUniform/layer_3/weights3/Initializer/random_uniform/sub*
T0*#
_class
loc:@layer_3/weights3*
_output_shapes

:
�
+layer_3/weights3/Initializer/random_uniformAdd/layer_3/weights3/Initializer/random_uniform/mul/layer_3/weights3/Initializer/random_uniform/min*
_output_shapes

:*
T0*#
_class
loc:@layer_3/weights3
�
layer_3/weights3
VariableV2*#
_class
loc:@layer_3/weights3*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
�
layer_3/weights3/AssignAssignlayer_3/weights3+layer_3/weights3/Initializer/random_uniform*
use_locking(*
T0*#
_class
loc:@layer_3/weights3*
validate_shape(*
_output_shapes

:
�
layer_3/weights3/readIdentitylayer_3/weights3*
T0*#
_class
loc:@layer_3/weights3*
_output_shapes

:
�
!layer_3/biases3/Initializer/zerosConst*"
_class
loc:@layer_3/biases3*
valueB*    *
dtype0*
_output_shapes
:
�
layer_3/biases3
VariableV2*
shared_name *"
_class
loc:@layer_3/biases3*
	container *
shape:*
dtype0*
_output_shapes
:
�
layer_3/biases3/AssignAssignlayer_3/biases3!layer_3/biases3/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@layer_3/biases3*
validate_shape(*
_output_shapes
:
z
layer_3/biases3/readIdentitylayer_3/biases3*
T0*"
_class
loc:@layer_3/biases3*
_output_shapes
:
�
layer_3/MatMulMatMullayer_2/Relulayer_3/weights3/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
j
layer_3/addAddlayer_3/MatMullayer_3/biases3/read*
T0*'
_output_shapes
:���������
S
layer_3/ReluRelulayer_3/add*
T0*'
_output_shapes
:���������
�
0output/weights4/Initializer/random_uniform/shapeConst*"
_class
loc:@output/weights4*
valueB"      *
dtype0*
_output_shapes
:
�
.output/weights4/Initializer/random_uniform/minConst*"
_class
loc:@output/weights4*
valueB
 *qĜ�*
dtype0*
_output_shapes
: 
�
.output/weights4/Initializer/random_uniform/maxConst*"
_class
loc:@output/weights4*
valueB
 *qĜ?*
dtype0*
_output_shapes
: 
�
8output/weights4/Initializer/random_uniform/RandomUniformRandomUniform0output/weights4/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:*

seed *
T0*"
_class
loc:@output/weights4*
seed2 
�
.output/weights4/Initializer/random_uniform/subSub.output/weights4/Initializer/random_uniform/max.output/weights4/Initializer/random_uniform/min*
_output_shapes
: *
T0*"
_class
loc:@output/weights4
�
.output/weights4/Initializer/random_uniform/mulMul8output/weights4/Initializer/random_uniform/RandomUniform.output/weights4/Initializer/random_uniform/sub*
_output_shapes

:*
T0*"
_class
loc:@output/weights4
�
*output/weights4/Initializer/random_uniformAdd.output/weights4/Initializer/random_uniform/mul.output/weights4/Initializer/random_uniform/min*
T0*"
_class
loc:@output/weights4*
_output_shapes

:
�
output/weights4
VariableV2*
dtype0*
_output_shapes

:*
shared_name *"
_class
loc:@output/weights4*
	container *
shape
:
�
output/weights4/AssignAssignoutput/weights4*output/weights4/Initializer/random_uniform*
T0*"
_class
loc:@output/weights4*
validate_shape(*
_output_shapes

:*
use_locking(
~
output/weights4/readIdentityoutput/weights4*
T0*"
_class
loc:@output/weights4*
_output_shapes

:
�
 output/biases4/Initializer/zerosConst*!
_class
loc:@output/biases4*
valueB*    *
dtype0*
_output_shapes
:
�
output/biases4
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *!
_class
loc:@output/biases4
�
output/biases4/AssignAssignoutput/biases4 output/biases4/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@output/biases4*
validate_shape(*
_output_shapes
:
w
output/biases4/readIdentityoutput/biases4*
T0*!
_class
loc:@output/biases4*
_output_shapes
:
�
output/MatMulMatMullayer_3/Reluoutput/weights4/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
g

output/addAddoutput/MatMuloutput/biases4/read*
T0*'
_output_shapes
:���������
s
cost/PlaceholderPlaceholder*
dtype0*'
_output_shapes
:���������*
shape:���������
{
cost/SquaredDifferenceSquaredDifference
output/addcost/Placeholder*
T0*'
_output_shapes
:���������
[

cost/ConstConst*
dtype0*
_output_shapes
:*
valueB"       
s
	cost/MeanMeancost/SquaredDifference
cost/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
X
train/gradients/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
^
train/gradients/grad_ys_0Const*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
train/gradients/FillFilltrain/gradients/Shapetrain/gradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
}
,train/gradients/cost/Mean_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
�
&train/gradients/cost/Mean_grad/ReshapeReshapetrain/gradients/Fill,train/gradients/cost/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
z
$train/gradients/cost/Mean_grad/ShapeShapecost/SquaredDifference*
T0*
out_type0*
_output_shapes
:
�
#train/gradients/cost/Mean_grad/TileTile&train/gradients/cost/Mean_grad/Reshape$train/gradients/cost/Mean_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:���������
|
&train/gradients/cost/Mean_grad/Shape_1Shapecost/SquaredDifference*
T0*
out_type0*
_output_shapes
:
i
&train/gradients/cost/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
n
$train/gradients/cost/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
#train/gradients/cost/Mean_grad/ProdProd&train/gradients/cost/Mean_grad/Shape_1$train/gradients/cost/Mean_grad/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
p
&train/gradients/cost/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
%train/gradients/cost/Mean_grad/Prod_1Prod&train/gradients/cost/Mean_grad/Shape_2&train/gradients/cost/Mean_grad/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
j
(train/gradients/cost/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
&train/gradients/cost/Mean_grad/MaximumMaximum%train/gradients/cost/Mean_grad/Prod_1(train/gradients/cost/Mean_grad/Maximum/y*
_output_shapes
: *
T0
�
'train/gradients/cost/Mean_grad/floordivFloorDiv#train/gradients/cost/Mean_grad/Prod&train/gradients/cost/Mean_grad/Maximum*
T0*
_output_shapes
: 
�
#train/gradients/cost/Mean_grad/CastCast'train/gradients/cost/Mean_grad/floordiv*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
�
&train/gradients/cost/Mean_grad/truedivRealDiv#train/gradients/cost/Mean_grad/Tile#train/gradients/cost/Mean_grad/Cast*
T0*'
_output_shapes
:���������
{
1train/gradients/cost/SquaredDifference_grad/ShapeShape
output/add*
T0*
out_type0*
_output_shapes
:
�
3train/gradients/cost/SquaredDifference_grad/Shape_1Shapecost/Placeholder*
T0*
out_type0*
_output_shapes
:
�
Atrain/gradients/cost/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgs1train/gradients/cost/SquaredDifference_grad/Shape3train/gradients/cost/SquaredDifference_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
2train/gradients/cost/SquaredDifference_grad/scalarConst'^train/gradients/cost/Mean_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
�
/train/gradients/cost/SquaredDifference_grad/mulMul2train/gradients/cost/SquaredDifference_grad/scalar&train/gradients/cost/Mean_grad/truediv*
T0*'
_output_shapes
:���������
�
/train/gradients/cost/SquaredDifference_grad/subSub
output/addcost/Placeholder'^train/gradients/cost/Mean_grad/truediv*'
_output_shapes
:���������*
T0
�
1train/gradients/cost/SquaredDifference_grad/mul_1Mul/train/gradients/cost/SquaredDifference_grad/mul/train/gradients/cost/SquaredDifference_grad/sub*
T0*'
_output_shapes
:���������
�
/train/gradients/cost/SquaredDifference_grad/SumSum1train/gradients/cost/SquaredDifference_grad/mul_1Atrain/gradients/cost/SquaredDifference_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
3train/gradients/cost/SquaredDifference_grad/ReshapeReshape/train/gradients/cost/SquaredDifference_grad/Sum1train/gradients/cost/SquaredDifference_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
1train/gradients/cost/SquaredDifference_grad/Sum_1Sum1train/gradients/cost/SquaredDifference_grad/mul_1Ctrain/gradients/cost/SquaredDifference_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
5train/gradients/cost/SquaredDifference_grad/Reshape_1Reshape1train/gradients/cost/SquaredDifference_grad/Sum_13train/gradients/cost/SquaredDifference_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
�
/train/gradients/cost/SquaredDifference_grad/NegNeg5train/gradients/cost/SquaredDifference_grad/Reshape_1*
T0*'
_output_shapes
:���������
�
<train/gradients/cost/SquaredDifference_grad/tuple/group_depsNoOp0^train/gradients/cost/SquaredDifference_grad/Neg4^train/gradients/cost/SquaredDifference_grad/Reshape
�
Dtrain/gradients/cost/SquaredDifference_grad/tuple/control_dependencyIdentity3train/gradients/cost/SquaredDifference_grad/Reshape=^train/gradients/cost/SquaredDifference_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*F
_class<
:8loc:@train/gradients/cost/SquaredDifference_grad/Reshape
�
Ftrain/gradients/cost/SquaredDifference_grad/tuple/control_dependency_1Identity/train/gradients/cost/SquaredDifference_grad/Neg=^train/gradients/cost/SquaredDifference_grad/tuple/group_deps*
T0*B
_class8
64loc:@train/gradients/cost/SquaredDifference_grad/Neg*'
_output_shapes
:���������
r
%train/gradients/output/add_grad/ShapeShapeoutput/MatMul*
T0*
out_type0*
_output_shapes
:
q
'train/gradients/output/add_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
�
5train/gradients/output/add_grad/BroadcastGradientArgsBroadcastGradientArgs%train/gradients/output/add_grad/Shape'train/gradients/output/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
#train/gradients/output/add_grad/SumSumDtrain/gradients/cost/SquaredDifference_grad/tuple/control_dependency5train/gradients/output/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
'train/gradients/output/add_grad/ReshapeReshape#train/gradients/output/add_grad/Sum%train/gradients/output/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
%train/gradients/output/add_grad/Sum_1SumDtrain/gradients/cost/SquaredDifference_grad/tuple/control_dependency7train/gradients/output/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
)train/gradients/output/add_grad/Reshape_1Reshape%train/gradients/output/add_grad/Sum_1'train/gradients/output/add_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
�
0train/gradients/output/add_grad/tuple/group_depsNoOp(^train/gradients/output/add_grad/Reshape*^train/gradients/output/add_grad/Reshape_1
�
8train/gradients/output/add_grad/tuple/control_dependencyIdentity'train/gradients/output/add_grad/Reshape1^train/gradients/output/add_grad/tuple/group_deps*
T0*:
_class0
.,loc:@train/gradients/output/add_grad/Reshape*'
_output_shapes
:���������
�
:train/gradients/output/add_grad/tuple/control_dependency_1Identity)train/gradients/output/add_grad/Reshape_11^train/gradients/output/add_grad/tuple/group_deps*
T0*<
_class2
0.loc:@train/gradients/output/add_grad/Reshape_1*
_output_shapes
:
�
)train/gradients/output/MatMul_grad/MatMulMatMul8train/gradients/output/add_grad/tuple/control_dependencyoutput/weights4/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b(
�
+train/gradients/output/MatMul_grad/MatMul_1MatMullayer_3/Relu8train/gradients/output/add_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:*
transpose_a(
�
3train/gradients/output/MatMul_grad/tuple/group_depsNoOp*^train/gradients/output/MatMul_grad/MatMul,^train/gradients/output/MatMul_grad/MatMul_1
�
;train/gradients/output/MatMul_grad/tuple/control_dependencyIdentity)train/gradients/output/MatMul_grad/MatMul4^train/gradients/output/MatMul_grad/tuple/group_deps*
T0*<
_class2
0.loc:@train/gradients/output/MatMul_grad/MatMul*'
_output_shapes
:���������
�
=train/gradients/output/MatMul_grad/tuple/control_dependency_1Identity+train/gradients/output/MatMul_grad/MatMul_14^train/gradients/output/MatMul_grad/tuple/group_deps*
T0*>
_class4
20loc:@train/gradients/output/MatMul_grad/MatMul_1*
_output_shapes

:
�
*train/gradients/layer_3/Relu_grad/ReluGradReluGrad;train/gradients/output/MatMul_grad/tuple/control_dependencylayer_3/Relu*
T0*'
_output_shapes
:���������
t
&train/gradients/layer_3/add_grad/ShapeShapelayer_3/MatMul*
T0*
out_type0*
_output_shapes
:
r
(train/gradients/layer_3/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
6train/gradients/layer_3/add_grad/BroadcastGradientArgsBroadcastGradientArgs&train/gradients/layer_3/add_grad/Shape(train/gradients/layer_3/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
$train/gradients/layer_3/add_grad/SumSum*train/gradients/layer_3/Relu_grad/ReluGrad6train/gradients/layer_3/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
(train/gradients/layer_3/add_grad/ReshapeReshape$train/gradients/layer_3/add_grad/Sum&train/gradients/layer_3/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
&train/gradients/layer_3/add_grad/Sum_1Sum*train/gradients/layer_3/Relu_grad/ReluGrad8train/gradients/layer_3/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
*train/gradients/layer_3/add_grad/Reshape_1Reshape&train/gradients/layer_3/add_grad/Sum_1(train/gradients/layer_3/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
�
1train/gradients/layer_3/add_grad/tuple/group_depsNoOp)^train/gradients/layer_3/add_grad/Reshape+^train/gradients/layer_3/add_grad/Reshape_1
�
9train/gradients/layer_3/add_grad/tuple/control_dependencyIdentity(train/gradients/layer_3/add_grad/Reshape2^train/gradients/layer_3/add_grad/tuple/group_deps*
T0*;
_class1
/-loc:@train/gradients/layer_3/add_grad/Reshape*'
_output_shapes
:���������
�
;train/gradients/layer_3/add_grad/tuple/control_dependency_1Identity*train/gradients/layer_3/add_grad/Reshape_12^train/gradients/layer_3/add_grad/tuple/group_deps*
_output_shapes
:*
T0*=
_class3
1/loc:@train/gradients/layer_3/add_grad/Reshape_1
�
*train/gradients/layer_3/MatMul_grad/MatMulMatMul9train/gradients/layer_3/add_grad/tuple/control_dependencylayer_3/weights3/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b(
�
,train/gradients/layer_3/MatMul_grad/MatMul_1MatMullayer_2/Relu9train/gradients/layer_3/add_grad/tuple/control_dependency*
_output_shapes

:*
transpose_a(*
transpose_b( *
T0
�
4train/gradients/layer_3/MatMul_grad/tuple/group_depsNoOp+^train/gradients/layer_3/MatMul_grad/MatMul-^train/gradients/layer_3/MatMul_grad/MatMul_1
�
<train/gradients/layer_3/MatMul_grad/tuple/control_dependencyIdentity*train/gradients/layer_3/MatMul_grad/MatMul5^train/gradients/layer_3/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_3/MatMul_grad/MatMul*'
_output_shapes
:���������
�
>train/gradients/layer_3/MatMul_grad/tuple/control_dependency_1Identity,train/gradients/layer_3/MatMul_grad/MatMul_15^train/gradients/layer_3/MatMul_grad/tuple/group_deps*
_output_shapes

:*
T0*?
_class5
31loc:@train/gradients/layer_3/MatMul_grad/MatMul_1
�
*train/gradients/layer_2/Relu_grad/ReluGradReluGrad<train/gradients/layer_3/MatMul_grad/tuple/control_dependencylayer_2/Relu*
T0*'
_output_shapes
:���������
t
&train/gradients/layer_2/add_grad/ShapeShapelayer_2/MatMul*
T0*
out_type0*
_output_shapes
:
r
(train/gradients/layer_2/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
6train/gradients/layer_2/add_grad/BroadcastGradientArgsBroadcastGradientArgs&train/gradients/layer_2/add_grad/Shape(train/gradients/layer_2/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
$train/gradients/layer_2/add_grad/SumSum*train/gradients/layer_2/Relu_grad/ReluGrad6train/gradients/layer_2/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
(train/gradients/layer_2/add_grad/ReshapeReshape$train/gradients/layer_2/add_grad/Sum&train/gradients/layer_2/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
&train/gradients/layer_2/add_grad/Sum_1Sum*train/gradients/layer_2/Relu_grad/ReluGrad8train/gradients/layer_2/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
*train/gradients/layer_2/add_grad/Reshape_1Reshape&train/gradients/layer_2/add_grad/Sum_1(train/gradients/layer_2/add_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
�
1train/gradients/layer_2/add_grad/tuple/group_depsNoOp)^train/gradients/layer_2/add_grad/Reshape+^train/gradients/layer_2/add_grad/Reshape_1
�
9train/gradients/layer_2/add_grad/tuple/control_dependencyIdentity(train/gradients/layer_2/add_grad/Reshape2^train/gradients/layer_2/add_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*;
_class1
/-loc:@train/gradients/layer_2/add_grad/Reshape
�
;train/gradients/layer_2/add_grad/tuple/control_dependency_1Identity*train/gradients/layer_2/add_grad/Reshape_12^train/gradients/layer_2/add_grad/tuple/group_deps*
_output_shapes
:*
T0*=
_class3
1/loc:@train/gradients/layer_2/add_grad/Reshape_1
�
*train/gradients/layer_2/MatMul_grad/MatMulMatMul9train/gradients/layer_2/add_grad/tuple/control_dependencylayer_2/weights2/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b(
�
,train/gradients/layer_2/MatMul_grad/MatMul_1MatMullayer_1/Relu9train/gradients/layer_2/add_grad/tuple/control_dependency*
_output_shapes

:*
transpose_a(*
transpose_b( *
T0
�
4train/gradients/layer_2/MatMul_grad/tuple/group_depsNoOp+^train/gradients/layer_2/MatMul_grad/MatMul-^train/gradients/layer_2/MatMul_grad/MatMul_1
�
<train/gradients/layer_2/MatMul_grad/tuple/control_dependencyIdentity*train/gradients/layer_2/MatMul_grad/MatMul5^train/gradients/layer_2/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_2/MatMul_grad/MatMul*'
_output_shapes
:���������
�
>train/gradients/layer_2/MatMul_grad/tuple/control_dependency_1Identity,train/gradients/layer_2/MatMul_grad/MatMul_15^train/gradients/layer_2/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@train/gradients/layer_2/MatMul_grad/MatMul_1*
_output_shapes

:
�
*train/gradients/layer_1/Relu_grad/ReluGradReluGrad<train/gradients/layer_2/MatMul_grad/tuple/control_dependencylayer_1/Relu*
T0*'
_output_shapes
:���������
t
&train/gradients/layer_1/add_grad/ShapeShapelayer_1/MatMul*
T0*
out_type0*
_output_shapes
:
r
(train/gradients/layer_1/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
6train/gradients/layer_1/add_grad/BroadcastGradientArgsBroadcastGradientArgs&train/gradients/layer_1/add_grad/Shape(train/gradients/layer_1/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
$train/gradients/layer_1/add_grad/SumSum*train/gradients/layer_1/Relu_grad/ReluGrad6train/gradients/layer_1/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
(train/gradients/layer_1/add_grad/ReshapeReshape$train/gradients/layer_1/add_grad/Sum&train/gradients/layer_1/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
&train/gradients/layer_1/add_grad/Sum_1Sum*train/gradients/layer_1/Relu_grad/ReluGrad8train/gradients/layer_1/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
*train/gradients/layer_1/add_grad/Reshape_1Reshape&train/gradients/layer_1/add_grad/Sum_1(train/gradients/layer_1/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
�
1train/gradients/layer_1/add_grad/tuple/group_depsNoOp)^train/gradients/layer_1/add_grad/Reshape+^train/gradients/layer_1/add_grad/Reshape_1
�
9train/gradients/layer_1/add_grad/tuple/control_dependencyIdentity(train/gradients/layer_1/add_grad/Reshape2^train/gradients/layer_1/add_grad/tuple/group_deps*
T0*;
_class1
/-loc:@train/gradients/layer_1/add_grad/Reshape*'
_output_shapes
:���������
�
;train/gradients/layer_1/add_grad/tuple/control_dependency_1Identity*train/gradients/layer_1/add_grad/Reshape_12^train/gradients/layer_1/add_grad/tuple/group_deps*
_output_shapes
:*
T0*=
_class3
1/loc:@train/gradients/layer_1/add_grad/Reshape_1
�
*train/gradients/layer_1/MatMul_grad/MatMulMatMul9train/gradients/layer_1/add_grad/tuple/control_dependencylayer_1/weights1/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b(*
T0
�
,train/gradients/layer_1/MatMul_grad/MatMul_1MatMulinput/Placeholder9train/gradients/layer_1/add_grad/tuple/control_dependency*
T0*
_output_shapes

:*
transpose_a(*
transpose_b( 
�
4train/gradients/layer_1/MatMul_grad/tuple/group_depsNoOp+^train/gradients/layer_1/MatMul_grad/MatMul-^train/gradients/layer_1/MatMul_grad/MatMul_1
�
<train/gradients/layer_1/MatMul_grad/tuple/control_dependencyIdentity*train/gradients/layer_1/MatMul_grad/MatMul5^train/gradients/layer_1/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_1/MatMul_grad/MatMul*'
_output_shapes
:���������
�
>train/gradients/layer_1/MatMul_grad/tuple/control_dependency_1Identity,train/gradients/layer_1/MatMul_grad/MatMul_15^train/gradients/layer_1/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@train/gradients/layer_1/MatMul_grad/MatMul_1*
_output_shapes

:
�
train/beta1_power/initial_valueConst*"
_class
loc:@layer_1/biases1*
valueB
 *fff?*
dtype0*
_output_shapes
: 
�
train/beta1_power
VariableV2*
dtype0*
_output_shapes
: *
shared_name *"
_class
loc:@layer_1/biases1*
	container *
shape: 
�
train/beta1_power/AssignAssigntrain/beta1_powertrain/beta1_power/initial_value*
use_locking(*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
: 
z
train/beta1_power/readIdentitytrain/beta1_power*
T0*"
_class
loc:@layer_1/biases1*
_output_shapes
: 
�
train/beta2_power/initial_valueConst*"
_class
loc:@layer_1/biases1*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
train/beta2_power
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *"
_class
loc:@layer_1/biases1*
	container 
�
train/beta2_power/AssignAssigntrain/beta2_powertrain/beta2_power/initial_value*
use_locking(*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
: 
z
train/beta2_power/readIdentitytrain/beta2_power*
T0*"
_class
loc:@layer_1/biases1*
_output_shapes
: 
�
-train/layer_1/weights1/Adam/Initializer/zerosConst*#
_class
loc:@layer_1/weights1*
valueB*    *
dtype0*
_output_shapes

:
�
train/layer_1/weights1/Adam
VariableV2*
shared_name *#
_class
loc:@layer_1/weights1*
	container *
shape
:*
dtype0*
_output_shapes

:
�
"train/layer_1/weights1/Adam/AssignAssigntrain/layer_1/weights1/Adam-train/layer_1/weights1/Adam/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@layer_1/weights1*
validate_shape(*
_output_shapes

:
�
 train/layer_1/weights1/Adam/readIdentitytrain/layer_1/weights1/Adam*
T0*#
_class
loc:@layer_1/weights1*
_output_shapes

:
�
/train/layer_1/weights1/Adam_1/Initializer/zerosConst*#
_class
loc:@layer_1/weights1*
valueB*    *
dtype0*
_output_shapes

:
�
train/layer_1/weights1/Adam_1
VariableV2*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@layer_1/weights1*
	container *
shape
:
�
$train/layer_1/weights1/Adam_1/AssignAssigntrain/layer_1/weights1/Adam_1/train/layer_1/weights1/Adam_1/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@layer_1/weights1*
validate_shape(*
_output_shapes

:
�
"train/layer_1/weights1/Adam_1/readIdentitytrain/layer_1/weights1/Adam_1*
_output_shapes

:*
T0*#
_class
loc:@layer_1/weights1
�
,train/layer_1/biases1/Adam/Initializer/zerosConst*"
_class
loc:@layer_1/biases1*
valueB*    *
dtype0*
_output_shapes
:
�
train/layer_1/biases1/Adam
VariableV2*"
_class
loc:@layer_1/biases1*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
�
!train/layer_1/biases1/Adam/AssignAssigntrain/layer_1/biases1/Adam,train/layer_1/biases1/Adam/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
:
�
train/layer_1/biases1/Adam/readIdentitytrain/layer_1/biases1/Adam*
_output_shapes
:*
T0*"
_class
loc:@layer_1/biases1
�
.train/layer_1/biases1/Adam_1/Initializer/zerosConst*"
_class
loc:@layer_1/biases1*
valueB*    *
dtype0*
_output_shapes
:
�
train/layer_1/biases1/Adam_1
VariableV2*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@layer_1/biases1*
	container *
shape:
�
#train/layer_1/biases1/Adam_1/AssignAssigntrain/layer_1/biases1/Adam_1.train/layer_1/biases1/Adam_1/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
:
�
!train/layer_1/biases1/Adam_1/readIdentitytrain/layer_1/biases1/Adam_1*
T0*"
_class
loc:@layer_1/biases1*
_output_shapes
:
�
-train/layer_2/weights2/Adam/Initializer/zerosConst*#
_class
loc:@layer_2/weights2*
valueB*    *
dtype0*
_output_shapes

:
�
train/layer_2/weights2/Adam
VariableV2*
shared_name *#
_class
loc:@layer_2/weights2*
	container *
shape
:*
dtype0*
_output_shapes

:
�
"train/layer_2/weights2/Adam/AssignAssigntrain/layer_2/weights2/Adam-train/layer_2/weights2/Adam/Initializer/zeros*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@layer_2/weights2
�
 train/layer_2/weights2/Adam/readIdentitytrain/layer_2/weights2/Adam*
T0*#
_class
loc:@layer_2/weights2*
_output_shapes

:
�
/train/layer_2/weights2/Adam_1/Initializer/zerosConst*#
_class
loc:@layer_2/weights2*
valueB*    *
dtype0*
_output_shapes

:
�
train/layer_2/weights2/Adam_1
VariableV2*#
_class
loc:@layer_2/weights2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
�
$train/layer_2/weights2/Adam_1/AssignAssigntrain/layer_2/weights2/Adam_1/train/layer_2/weights2/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@layer_2/weights2
�
"train/layer_2/weights2/Adam_1/readIdentitytrain/layer_2/weights2/Adam_1*
T0*#
_class
loc:@layer_2/weights2*
_output_shapes

:
�
,train/layer_2/biases2/Adam/Initializer/zerosConst*"
_class
loc:@layer_2/biases2*
valueB*    *
dtype0*
_output_shapes
:
�
train/layer_2/biases2/Adam
VariableV2*
shared_name *"
_class
loc:@layer_2/biases2*
	container *
shape:*
dtype0*
_output_shapes
:
�
!train/layer_2/biases2/Adam/AssignAssigntrain/layer_2/biases2/Adam,train/layer_2/biases2/Adam/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@layer_2/biases2*
validate_shape(*
_output_shapes
:
�
train/layer_2/biases2/Adam/readIdentitytrain/layer_2/biases2/Adam*
_output_shapes
:*
T0*"
_class
loc:@layer_2/biases2
�
.train/layer_2/biases2/Adam_1/Initializer/zerosConst*"
_class
loc:@layer_2/biases2*
valueB*    *
dtype0*
_output_shapes
:
�
train/layer_2/biases2/Adam_1
VariableV2*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@layer_2/biases2*
	container *
shape:
�
#train/layer_2/biases2/Adam_1/AssignAssigntrain/layer_2/biases2/Adam_1.train/layer_2/biases2/Adam_1/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@layer_2/biases2*
validate_shape(*
_output_shapes
:
�
!train/layer_2/biases2/Adam_1/readIdentitytrain/layer_2/biases2/Adam_1*
T0*"
_class
loc:@layer_2/biases2*
_output_shapes
:
�
-train/layer_3/weights3/Adam/Initializer/zerosConst*
dtype0*
_output_shapes

:*#
_class
loc:@layer_3/weights3*
valueB*    
�
train/layer_3/weights3/Adam
VariableV2*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@layer_3/weights3*
	container *
shape
:
�
"train/layer_3/weights3/Adam/AssignAssigntrain/layer_3/weights3/Adam-train/layer_3/weights3/Adam/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@layer_3/weights3*
validate_shape(*
_output_shapes

:
�
 train/layer_3/weights3/Adam/readIdentitytrain/layer_3/weights3/Adam*
_output_shapes

:*
T0*#
_class
loc:@layer_3/weights3
�
/train/layer_3/weights3/Adam_1/Initializer/zerosConst*#
_class
loc:@layer_3/weights3*
valueB*    *
dtype0*
_output_shapes

:
�
train/layer_3/weights3/Adam_1
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@layer_3/weights3*
	container 
�
$train/layer_3/weights3/Adam_1/AssignAssigntrain/layer_3/weights3/Adam_1/train/layer_3/weights3/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@layer_3/weights3
�
"train/layer_3/weights3/Adam_1/readIdentitytrain/layer_3/weights3/Adam_1*
T0*#
_class
loc:@layer_3/weights3*
_output_shapes

:
�
,train/layer_3/biases3/Adam/Initializer/zerosConst*"
_class
loc:@layer_3/biases3*
valueB*    *
dtype0*
_output_shapes
:
�
train/layer_3/biases3/Adam
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@layer_3/biases3*
	container 
�
!train/layer_3/biases3/Adam/AssignAssigntrain/layer_3/biases3/Adam,train/layer_3/biases3/Adam/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@layer_3/biases3*
validate_shape(*
_output_shapes
:
�
train/layer_3/biases3/Adam/readIdentitytrain/layer_3/biases3/Adam*
T0*"
_class
loc:@layer_3/biases3*
_output_shapes
:
�
.train/layer_3/biases3/Adam_1/Initializer/zerosConst*"
_class
loc:@layer_3/biases3*
valueB*    *
dtype0*
_output_shapes
:
�
train/layer_3/biases3/Adam_1
VariableV2*"
_class
loc:@layer_3/biases3*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
�
#train/layer_3/biases3/Adam_1/AssignAssigntrain/layer_3/biases3/Adam_1.train/layer_3/biases3/Adam_1/Initializer/zeros*
T0*"
_class
loc:@layer_3/biases3*
validate_shape(*
_output_shapes
:*
use_locking(
�
!train/layer_3/biases3/Adam_1/readIdentitytrain/layer_3/biases3/Adam_1*
T0*"
_class
loc:@layer_3/biases3*
_output_shapes
:
�
,train/output/weights4/Adam/Initializer/zerosConst*
dtype0*
_output_shapes

:*"
_class
loc:@output/weights4*
valueB*    
�
train/output/weights4/Adam
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name *"
_class
loc:@output/weights4
�
!train/output/weights4/Adam/AssignAssigntrain/output/weights4/Adam,train/output/weights4/Adam/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@output/weights4*
validate_shape(*
_output_shapes

:
�
train/output/weights4/Adam/readIdentitytrain/output/weights4/Adam*
T0*"
_class
loc:@output/weights4*
_output_shapes

:
�
.train/output/weights4/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes

:*"
_class
loc:@output/weights4*
valueB*    
�
train/output/weights4/Adam_1
VariableV2*"
_class
loc:@output/weights4*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
�
#train/output/weights4/Adam_1/AssignAssigntrain/output/weights4/Adam_1.train/output/weights4/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*"
_class
loc:@output/weights4
�
!train/output/weights4/Adam_1/readIdentitytrain/output/weights4/Adam_1*
T0*"
_class
loc:@output/weights4*
_output_shapes

:
�
+train/output/biases4/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:*!
_class
loc:@output/biases4*
valueB*    
�
train/output/biases4/Adam
VariableV2*
dtype0*
_output_shapes
:*
shared_name *!
_class
loc:@output/biases4*
	container *
shape:
�
 train/output/biases4/Adam/AssignAssigntrain/output/biases4/Adam+train/output/biases4/Adam/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@output/biases4*
validate_shape(*
_output_shapes
:
�
train/output/biases4/Adam/readIdentitytrain/output/biases4/Adam*
T0*!
_class
loc:@output/biases4*
_output_shapes
:
�
-train/output/biases4/Adam_1/Initializer/zerosConst*!
_class
loc:@output/biases4*
valueB*    *
dtype0*
_output_shapes
:
�
train/output/biases4/Adam_1
VariableV2*
dtype0*
_output_shapes
:*
shared_name *!
_class
loc:@output/biases4*
	container *
shape:
�
"train/output/biases4/Adam_1/AssignAssigntrain/output/biases4/Adam_1-train/output/biases4/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*!
_class
loc:@output/biases4
�
 train/output/biases4/Adam_1/readIdentitytrain/output/biases4/Adam_1*
T0*!
_class
loc:@output/biases4*
_output_shapes
:
]
train/Adam/learning_rateConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
U
train/Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
U
train/Adam/beta2Const*
dtype0*
_output_shapes
: *
valueB
 *w�?
W
train/Adam/epsilonConst*
dtype0*
_output_shapes
: *
valueB
 *w�+2
�
,train/Adam/update_layer_1/weights1/ApplyAdam	ApplyAdamlayer_1/weights1train/layer_1/weights1/Adamtrain/layer_1/weights1/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon>train/gradients/layer_1/MatMul_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes

:*
use_locking( *
T0*#
_class
loc:@layer_1/weights1
�
+train/Adam/update_layer_1/biases1/ApplyAdam	ApplyAdamlayer_1/biases1train/layer_1/biases1/Adamtrain/layer_1/biases1/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon;train/gradients/layer_1/add_grad/tuple/control_dependency_1*
use_locking( *
T0*"
_class
loc:@layer_1/biases1*
use_nesterov( *
_output_shapes
:
�
,train/Adam/update_layer_2/weights2/ApplyAdam	ApplyAdamlayer_2/weights2train/layer_2/weights2/Adamtrain/layer_2/weights2/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon>train/gradients/layer_2/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*#
_class
loc:@layer_2/weights2*
use_nesterov( *
_output_shapes

:
�
+train/Adam/update_layer_2/biases2/ApplyAdam	ApplyAdamlayer_2/biases2train/layer_2/biases2/Adamtrain/layer_2/biases2/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon;train/gradients/layer_2/add_grad/tuple/control_dependency_1*
T0*"
_class
loc:@layer_2/biases2*
use_nesterov( *
_output_shapes
:*
use_locking( 
�
,train/Adam/update_layer_3/weights3/ApplyAdam	ApplyAdamlayer_3/weights3train/layer_3/weights3/Adamtrain/layer_3/weights3/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon>train/gradients/layer_3/MatMul_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes

:*
use_locking( *
T0*#
_class
loc:@layer_3/weights3
�
+train/Adam/update_layer_3/biases3/ApplyAdam	ApplyAdamlayer_3/biases3train/layer_3/biases3/Adamtrain/layer_3/biases3/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon;train/gradients/layer_3/add_grad/tuple/control_dependency_1*
T0*"
_class
loc:@layer_3/biases3*
use_nesterov( *
_output_shapes
:*
use_locking( 
�
+train/Adam/update_output/weights4/ApplyAdam	ApplyAdamoutput/weights4train/output/weights4/Adamtrain/output/weights4/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon=train/gradients/output/MatMul_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes

:*
use_locking( *
T0*"
_class
loc:@output/weights4
�
*train/Adam/update_output/biases4/ApplyAdam	ApplyAdamoutput/biases4train/output/biases4/Adamtrain/output/biases4/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon:train/gradients/output/add_grad/tuple/control_dependency_1*
T0*!
_class
loc:@output/biases4*
use_nesterov( *
_output_shapes
:*
use_locking( 
�
train/Adam/mulMultrain/beta1_power/readtrain/Adam/beta1,^train/Adam/update_layer_1/biases1/ApplyAdam-^train/Adam/update_layer_1/weights1/ApplyAdam,^train/Adam/update_layer_2/biases2/ApplyAdam-^train/Adam/update_layer_2/weights2/ApplyAdam,^train/Adam/update_layer_3/biases3/ApplyAdam-^train/Adam/update_layer_3/weights3/ApplyAdam+^train/Adam/update_output/biases4/ApplyAdam,^train/Adam/update_output/weights4/ApplyAdam*
T0*"
_class
loc:@layer_1/biases1*
_output_shapes
: 
�
train/Adam/AssignAssigntrain/beta1_powertrain/Adam/mul*
use_locking( *
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
: 
�
train/Adam/mul_1Multrain/beta2_power/readtrain/Adam/beta2,^train/Adam/update_layer_1/biases1/ApplyAdam-^train/Adam/update_layer_1/weights1/ApplyAdam,^train/Adam/update_layer_2/biases2/ApplyAdam-^train/Adam/update_layer_2/weights2/ApplyAdam,^train/Adam/update_layer_3/biases3/ApplyAdam-^train/Adam/update_layer_3/weights3/ApplyAdam+^train/Adam/update_output/biases4/ApplyAdam,^train/Adam/update_output/weights4/ApplyAdam*
T0*"
_class
loc:@layer_1/biases1*
_output_shapes
: 
�
train/Adam/Assign_1Assigntrain/beta2_powertrain/Adam/mul_1*
validate_shape(*
_output_shapes
: *
use_locking( *
T0*"
_class
loc:@layer_1/biases1
�

train/AdamNoOp^train/Adam/Assign^train/Adam/Assign_1,^train/Adam/update_layer_1/biases1/ApplyAdam-^train/Adam/update_layer_1/weights1/ApplyAdam,^train/Adam/update_layer_2/biases2/ApplyAdam-^train/Adam/update_layer_2/weights2/ApplyAdam,^train/Adam/update_layer_3/biases3/ApplyAdam-^train/Adam/update_layer_3/weights3/ApplyAdam+^train/Adam/update_output/biases4/ApplyAdam,^train/Adam/update_output/weights4/ApplyAdam
n
logging/current_cost/tagsConst*%
valueB Blogging/current_cost*
dtype0*
_output_shapes
: 
l
logging/current_costScalarSummarylogging/current_cost/tags	cost/Mean*
T0*
_output_shapes
: 
a
logging/Merge/MergeSummaryMergeSummarylogging/current_cost*
N*
_output_shapes
: 
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
�
save/SaveV2/tensor_namesConst*�
value�B�Blayer_1/biases1Blayer_1/weights1Blayer_2/biases2Blayer_2/weights2Blayer_3/biases3Blayer_3/weights3Boutput/biases4Boutput/weights4Btrain/beta1_powerBtrain/beta2_powerBtrain/layer_1/biases1/AdamBtrain/layer_1/biases1/Adam_1Btrain/layer_1/weights1/AdamBtrain/layer_1/weights1/Adam_1Btrain/layer_2/biases2/AdamBtrain/layer_2/biases2/Adam_1Btrain/layer_2/weights2/AdamBtrain/layer_2/weights2/Adam_1Btrain/layer_3/biases3/AdamBtrain/layer_3/biases3/Adam_1Btrain/layer_3/weights3/AdamBtrain/layer_3/weights3/Adam_1Btrain/output/biases4/AdamBtrain/output/biases4/Adam_1Btrain/output/weights4/AdamBtrain/output/weights4/Adam_1*
dtype0*
_output_shapes
:
�
save/SaveV2/shape_and_slicesConst*
dtype0*
_output_shapes
:*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B 
�
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_sliceslayer_1/biases1layer_1/weights1layer_2/biases2layer_2/weights2layer_3/biases3layer_3/weights3output/biases4output/weights4train/beta1_powertrain/beta2_powertrain/layer_1/biases1/Adamtrain/layer_1/biases1/Adam_1train/layer_1/weights1/Adamtrain/layer_1/weights1/Adam_1train/layer_2/biases2/Adamtrain/layer_2/biases2/Adam_1train/layer_2/weights2/Adamtrain/layer_2/weights2/Adam_1train/layer_3/biases3/Adamtrain/layer_3/biases3/Adam_1train/layer_3/weights3/Adamtrain/layer_3/weights3/Adam_1train/output/biases4/Adamtrain/output/biases4/Adam_1train/output/weights4/Adamtrain/output/weights4/Adam_1*(
dtypes
2
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const*
_output_shapes
: 
�
save/RestoreV2/tensor_namesConst"/device:CPU:0*�
value�B�Blayer_1/biases1Blayer_1/weights1Blayer_2/biases2Blayer_2/weights2Blayer_3/biases3Blayer_3/weights3Boutput/biases4Boutput/weights4Btrain/beta1_powerBtrain/beta2_powerBtrain/layer_1/biases1/AdamBtrain/layer_1/biases1/Adam_1Btrain/layer_1/weights1/AdamBtrain/layer_1/weights1/Adam_1Btrain/layer_2/biases2/AdamBtrain/layer_2/biases2/Adam_1Btrain/layer_2/weights2/AdamBtrain/layer_2/weights2/Adam_1Btrain/layer_3/biases3/AdamBtrain/layer_3/biases3/Adam_1Btrain/layer_3/weights3/AdamBtrain/layer_3/weights3/Adam_1Btrain/output/biases4/AdamBtrain/output/biases4/Adam_1Btrain/output/weights4/AdamBtrain/output/weights4/Adam_1*
dtype0*
_output_shapes
:
�
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
�
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*|
_output_shapesj
h::::::::::::::::::::::::::*(
dtypes
2
�
save/AssignAssignlayer_1/biases1save/RestoreV2*
use_locking(*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
:
�
save/Assign_1Assignlayer_1/weights1save/RestoreV2:1*
use_locking(*
T0*#
_class
loc:@layer_1/weights1*
validate_shape(*
_output_shapes

:
�
save/Assign_2Assignlayer_2/biases2save/RestoreV2:2*
use_locking(*
T0*"
_class
loc:@layer_2/biases2*
validate_shape(*
_output_shapes
:
�
save/Assign_3Assignlayer_2/weights2save/RestoreV2:3*
T0*#
_class
loc:@layer_2/weights2*
validate_shape(*
_output_shapes

:*
use_locking(
�
save/Assign_4Assignlayer_3/biases3save/RestoreV2:4*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@layer_3/biases3
�
save/Assign_5Assignlayer_3/weights3save/RestoreV2:5*
use_locking(*
T0*#
_class
loc:@layer_3/weights3*
validate_shape(*
_output_shapes

:
�
save/Assign_6Assignoutput/biases4save/RestoreV2:6*
use_locking(*
T0*!
_class
loc:@output/biases4*
validate_shape(*
_output_shapes
:
�
save/Assign_7Assignoutput/weights4save/RestoreV2:7*
T0*"
_class
loc:@output/weights4*
validate_shape(*
_output_shapes

:*
use_locking(
�
save/Assign_8Assigntrain/beta1_powersave/RestoreV2:8*
use_locking(*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
: 
�
save/Assign_9Assigntrain/beta2_powersave/RestoreV2:9*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*"
_class
loc:@layer_1/biases1
�
save/Assign_10Assigntrain/layer_1/biases1/Adamsave/RestoreV2:10*
use_locking(*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
:
�
save/Assign_11Assigntrain/layer_1/biases1/Adam_1save/RestoreV2:11*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@layer_1/biases1
�
save/Assign_12Assigntrain/layer_1/weights1/Adamsave/RestoreV2:12*
use_locking(*
T0*#
_class
loc:@layer_1/weights1*
validate_shape(*
_output_shapes

:
�
save/Assign_13Assigntrain/layer_1/weights1/Adam_1save/RestoreV2:13*
T0*#
_class
loc:@layer_1/weights1*
validate_shape(*
_output_shapes

:*
use_locking(
�
save/Assign_14Assigntrain/layer_2/biases2/Adamsave/RestoreV2:14*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@layer_2/biases2
�
save/Assign_15Assigntrain/layer_2/biases2/Adam_1save/RestoreV2:15*
use_locking(*
T0*"
_class
loc:@layer_2/biases2*
validate_shape(*
_output_shapes
:
�
save/Assign_16Assigntrain/layer_2/weights2/Adamsave/RestoreV2:16*
use_locking(*
T0*#
_class
loc:@layer_2/weights2*
validate_shape(*
_output_shapes

:
�
save/Assign_17Assigntrain/layer_2/weights2/Adam_1save/RestoreV2:17*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@layer_2/weights2
�
save/Assign_18Assigntrain/layer_3/biases3/Adamsave/RestoreV2:18*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@layer_3/biases3
�
save/Assign_19Assigntrain/layer_3/biases3/Adam_1save/RestoreV2:19*
use_locking(*
T0*"
_class
loc:@layer_3/biases3*
validate_shape(*
_output_shapes
:
�
save/Assign_20Assigntrain/layer_3/weights3/Adamsave/RestoreV2:20*
use_locking(*
T0*#
_class
loc:@layer_3/weights3*
validate_shape(*
_output_shapes

:
�
save/Assign_21Assigntrain/layer_3/weights3/Adam_1save/RestoreV2:21*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@layer_3/weights3
�
save/Assign_22Assigntrain/output/biases4/Adamsave/RestoreV2:22*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*!
_class
loc:@output/biases4
�
save/Assign_23Assigntrain/output/biases4/Adam_1save/RestoreV2:23*
use_locking(*
T0*!
_class
loc:@output/biases4*
validate_shape(*
_output_shapes
:
�
save/Assign_24Assigntrain/output/weights4/Adamsave/RestoreV2:24*
use_locking(*
T0*"
_class
loc:@output/weights4*
validate_shape(*
_output_shapes

:
�
save/Assign_25Assigntrain/output/weights4/Adam_1save/RestoreV2:25*
use_locking(*
T0*"
_class
loc:@output/weights4*
validate_shape(*
_output_shapes

:
�
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
�
initNoOp^layer_1/biases1/Assign^layer_1/weights1/Assign^layer_2/biases2/Assign^layer_2/weights2/Assign^layer_3/biases3/Assign^layer_3/weights3/Assign^output/biases4/Assign^output/weights4/Assign^train/beta1_power/Assign^train/beta2_power/Assign"^train/layer_1/biases1/Adam/Assign$^train/layer_1/biases1/Adam_1/Assign#^train/layer_1/weights1/Adam/Assign%^train/layer_1/weights1/Adam_1/Assign"^train/layer_2/biases2/Adam/Assign$^train/layer_2/biases2/Adam_1/Assign#^train/layer_2/weights2/Adam/Assign%^train/layer_2/weights2/Adam_1/Assign"^train/layer_3/biases3/Adam/Assign$^train/layer_3/biases3/Adam_1/Assign#^train/layer_3/weights3/Adam/Assign%^train/layer_3/weights3/Adam_1/Assign!^train/output/biases4/Adam/Assign#^train/output/biases4/Adam_1/Assign"^train/output/weights4/Adam/Assign$^train/output/weights4/Adam_1/Assign"�ݏ�     ��d]	@O0��AJ܉
��
:
Add
x"T
y"T
z"T"
Ttype:
2	
�
	ApplyAdam
var"T�	
m"T�	
v"T�
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T�" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
x
Assign
ref"T�

value"T

output_ref"T�"	
Ttype"
validate_shapebool("
use_lockingbool(�
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2
;
Maximum
x"T
y"T
z"T"
Ttype:

2	�
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
8
MergeSummary
inputs*N
summary"
Nint(0
=
Mul
x"T
y"T
z"T"
Ttype:
2	�
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
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
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
P
ScalarSummary
tags
values"T
summary"
Ttype:
2	
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	�
:
Sub
x"T
y"T
z"T"
Ttype:
2	
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
s

VariableV2
ref"dtype�"
shapeshape"
dtypetype"
	containerstring "
shared_namestring �*1.12.02v1.12.0-0-ga6d8ffae09��
t
input/PlaceholderPlaceholder*
dtype0*'
_output_shapes
:���������*
shape:���������
�
1layer_1/weights1/Initializer/random_uniform/shapeConst*#
_class
loc:@layer_1/weights1*
valueB"      *
dtype0*
_output_shapes
:
�
/layer_1/weights1/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *#
_class
loc:@layer_1/weights1*
valueB
 *�7��
�
/layer_1/weights1/Initializer/random_uniform/maxConst*#
_class
loc:@layer_1/weights1*
valueB
 *�7�?*
dtype0*
_output_shapes
: 
�
9layer_1/weights1/Initializer/random_uniform/RandomUniformRandomUniform1layer_1/weights1/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:*

seed *
T0*#
_class
loc:@layer_1/weights1*
seed2 
�
/layer_1/weights1/Initializer/random_uniform/subSub/layer_1/weights1/Initializer/random_uniform/max/layer_1/weights1/Initializer/random_uniform/min*
T0*#
_class
loc:@layer_1/weights1*
_output_shapes
: 
�
/layer_1/weights1/Initializer/random_uniform/mulMul9layer_1/weights1/Initializer/random_uniform/RandomUniform/layer_1/weights1/Initializer/random_uniform/sub*
T0*#
_class
loc:@layer_1/weights1*
_output_shapes

:
�
+layer_1/weights1/Initializer/random_uniformAdd/layer_1/weights1/Initializer/random_uniform/mul/layer_1/weights1/Initializer/random_uniform/min*
_output_shapes

:*
T0*#
_class
loc:@layer_1/weights1
�
layer_1/weights1
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@layer_1/weights1
�
layer_1/weights1/AssignAssignlayer_1/weights1+layer_1/weights1/Initializer/random_uniform*
use_locking(*
T0*#
_class
loc:@layer_1/weights1*
validate_shape(*
_output_shapes

:
�
layer_1/weights1/readIdentitylayer_1/weights1*
T0*#
_class
loc:@layer_1/weights1*
_output_shapes

:
�
!layer_1/biases1/Initializer/zerosConst*
dtype0*
_output_shapes
:*"
_class
loc:@layer_1/biases1*
valueB*    
�
layer_1/biases1
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@layer_1/biases1*
	container 
�
layer_1/biases1/AssignAssignlayer_1/biases1!layer_1/biases1/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
:
z
layer_1/biases1/readIdentitylayer_1/biases1*
T0*"
_class
loc:@layer_1/biases1*
_output_shapes
:
�
layer_1/MatMulMatMulinput/Placeholderlayer_1/weights1/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
j
layer_1/addAddlayer_1/MatMullayer_1/biases1/read*'
_output_shapes
:���������*
T0
S
layer_1/ReluRelulayer_1/add*
T0*'
_output_shapes
:���������
�
1layer_2/weights2/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*#
_class
loc:@layer_2/weights2*
valueB"      
�
/layer_2/weights2/Initializer/random_uniform/minConst*#
_class
loc:@layer_2/weights2*
valueB
 *׳]�*
dtype0*
_output_shapes
: 
�
/layer_2/weights2/Initializer/random_uniform/maxConst*#
_class
loc:@layer_2/weights2*
valueB
 *׳]?*
dtype0*
_output_shapes
: 
�
9layer_2/weights2/Initializer/random_uniform/RandomUniformRandomUniform1layer_2/weights2/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:*

seed *
T0*#
_class
loc:@layer_2/weights2*
seed2 
�
/layer_2/weights2/Initializer/random_uniform/subSub/layer_2/weights2/Initializer/random_uniform/max/layer_2/weights2/Initializer/random_uniform/min*
T0*#
_class
loc:@layer_2/weights2*
_output_shapes
: 
�
/layer_2/weights2/Initializer/random_uniform/mulMul9layer_2/weights2/Initializer/random_uniform/RandomUniform/layer_2/weights2/Initializer/random_uniform/sub*
_output_shapes

:*
T0*#
_class
loc:@layer_2/weights2
�
+layer_2/weights2/Initializer/random_uniformAdd/layer_2/weights2/Initializer/random_uniform/mul/layer_2/weights2/Initializer/random_uniform/min*
T0*#
_class
loc:@layer_2/weights2*
_output_shapes

:
�
layer_2/weights2
VariableV2*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@layer_2/weights2*
	container *
shape
:
�
layer_2/weights2/AssignAssignlayer_2/weights2+layer_2/weights2/Initializer/random_uniform*
use_locking(*
T0*#
_class
loc:@layer_2/weights2*
validate_shape(*
_output_shapes

:
�
layer_2/weights2/readIdentitylayer_2/weights2*
T0*#
_class
loc:@layer_2/weights2*
_output_shapes

:
�
!layer_2/biases2/Initializer/zerosConst*"
_class
loc:@layer_2/biases2*
valueB*    *
dtype0*
_output_shapes
:
�
layer_2/biases2
VariableV2*
shared_name *"
_class
loc:@layer_2/biases2*
	container *
shape:*
dtype0*
_output_shapes
:
�
layer_2/biases2/AssignAssignlayer_2/biases2!layer_2/biases2/Initializer/zeros*
T0*"
_class
loc:@layer_2/biases2*
validate_shape(*
_output_shapes
:*
use_locking(
z
layer_2/biases2/readIdentitylayer_2/biases2*
_output_shapes
:*
T0*"
_class
loc:@layer_2/biases2
�
layer_2/MatMulMatMullayer_1/Relulayer_2/weights2/read*
transpose_b( *
T0*'
_output_shapes
:���������*
transpose_a( 
j
layer_2/addAddlayer_2/MatMullayer_2/biases2/read*
T0*'
_output_shapes
:���������
S
layer_2/ReluRelulayer_2/add*
T0*'
_output_shapes
:���������
�
1layer_3/weights3/Initializer/random_uniform/shapeConst*#
_class
loc:@layer_3/weights3*
valueB"      *
dtype0*
_output_shapes
:
�
/layer_3/weights3/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *#
_class
loc:@layer_3/weights3*
valueB
 *׳]�
�
/layer_3/weights3/Initializer/random_uniform/maxConst*#
_class
loc:@layer_3/weights3*
valueB
 *׳]?*
dtype0*
_output_shapes
: 
�
9layer_3/weights3/Initializer/random_uniform/RandomUniformRandomUniform1layer_3/weights3/Initializer/random_uniform/shape*
T0*#
_class
loc:@layer_3/weights3*
seed2 *
dtype0*
_output_shapes

:*

seed 
�
/layer_3/weights3/Initializer/random_uniform/subSub/layer_3/weights3/Initializer/random_uniform/max/layer_3/weights3/Initializer/random_uniform/min*
T0*#
_class
loc:@layer_3/weights3*
_output_shapes
: 
�
/layer_3/weights3/Initializer/random_uniform/mulMul9layer_3/weights3/Initializer/random_uniform/RandomUniform/layer_3/weights3/Initializer/random_uniform/sub*
T0*#
_class
loc:@layer_3/weights3*
_output_shapes

:
�
+layer_3/weights3/Initializer/random_uniformAdd/layer_3/weights3/Initializer/random_uniform/mul/layer_3/weights3/Initializer/random_uniform/min*
T0*#
_class
loc:@layer_3/weights3*
_output_shapes

:
�
layer_3/weights3
VariableV2*#
_class
loc:@layer_3/weights3*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
�
layer_3/weights3/AssignAssignlayer_3/weights3+layer_3/weights3/Initializer/random_uniform*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@layer_3/weights3
�
layer_3/weights3/readIdentitylayer_3/weights3*
T0*#
_class
loc:@layer_3/weights3*
_output_shapes

:
�
!layer_3/biases3/Initializer/zerosConst*"
_class
loc:@layer_3/biases3*
valueB*    *
dtype0*
_output_shapes
:
�
layer_3/biases3
VariableV2*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@layer_3/biases3*
	container *
shape:
�
layer_3/biases3/AssignAssignlayer_3/biases3!layer_3/biases3/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@layer_3/biases3*
validate_shape(*
_output_shapes
:
z
layer_3/biases3/readIdentitylayer_3/biases3*
_output_shapes
:*
T0*"
_class
loc:@layer_3/biases3
�
layer_3/MatMulMatMullayer_2/Relulayer_3/weights3/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
j
layer_3/addAddlayer_3/MatMullayer_3/biases3/read*
T0*'
_output_shapes
:���������
S
layer_3/ReluRelulayer_3/add*'
_output_shapes
:���������*
T0
�
0output/weights4/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*"
_class
loc:@output/weights4*
valueB"      
�
.output/weights4/Initializer/random_uniform/minConst*"
_class
loc:@output/weights4*
valueB
 *qĜ�*
dtype0*
_output_shapes
: 
�
.output/weights4/Initializer/random_uniform/maxConst*"
_class
loc:@output/weights4*
valueB
 *qĜ?*
dtype0*
_output_shapes
: 
�
8output/weights4/Initializer/random_uniform/RandomUniformRandomUniform0output/weights4/Initializer/random_uniform/shape*
seed2 *
dtype0*
_output_shapes

:*

seed *
T0*"
_class
loc:@output/weights4
�
.output/weights4/Initializer/random_uniform/subSub.output/weights4/Initializer/random_uniform/max.output/weights4/Initializer/random_uniform/min*
T0*"
_class
loc:@output/weights4*
_output_shapes
: 
�
.output/weights4/Initializer/random_uniform/mulMul8output/weights4/Initializer/random_uniform/RandomUniform.output/weights4/Initializer/random_uniform/sub*
T0*"
_class
loc:@output/weights4*
_output_shapes

:
�
*output/weights4/Initializer/random_uniformAdd.output/weights4/Initializer/random_uniform/mul.output/weights4/Initializer/random_uniform/min*
T0*"
_class
loc:@output/weights4*
_output_shapes

:
�
output/weights4
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name *"
_class
loc:@output/weights4
�
output/weights4/AssignAssignoutput/weights4*output/weights4/Initializer/random_uniform*
T0*"
_class
loc:@output/weights4*
validate_shape(*
_output_shapes

:*
use_locking(
~
output/weights4/readIdentityoutput/weights4*
T0*"
_class
loc:@output/weights4*
_output_shapes

:
�
 output/biases4/Initializer/zerosConst*!
_class
loc:@output/biases4*
valueB*    *
dtype0*
_output_shapes
:
�
output/biases4
VariableV2*
dtype0*
_output_shapes
:*
shared_name *!
_class
loc:@output/biases4*
	container *
shape:
�
output/biases4/AssignAssignoutput/biases4 output/biases4/Initializer/zeros*
T0*!
_class
loc:@output/biases4*
validate_shape(*
_output_shapes
:*
use_locking(
w
output/biases4/readIdentityoutput/biases4*
T0*!
_class
loc:@output/biases4*
_output_shapes
:
�
output/MatMulMatMullayer_3/Reluoutput/weights4/read*
transpose_b( *
T0*'
_output_shapes
:���������*
transpose_a( 
g

output/addAddoutput/MatMuloutput/biases4/read*'
_output_shapes
:���������*
T0
s
cost/PlaceholderPlaceholder*
shape:���������*
dtype0*'
_output_shapes
:���������
{
cost/SquaredDifferenceSquaredDifference
output/addcost/Placeholder*'
_output_shapes
:���������*
T0
[

cost/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
s
	cost/MeanMeancost/SquaredDifference
cost/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
X
train/gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
^
train/gradients/grad_ys_0Const*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
train/gradients/FillFilltrain/gradients/Shapetrain/gradients/grad_ys_0*
_output_shapes
: *
T0*

index_type0
}
,train/gradients/cost/Mean_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
�
&train/gradients/cost/Mean_grad/ReshapeReshapetrain/gradients/Fill,train/gradients/cost/Mean_grad/Reshape/shape*
_output_shapes

:*
T0*
Tshape0
z
$train/gradients/cost/Mean_grad/ShapeShapecost/SquaredDifference*
T0*
out_type0*
_output_shapes
:
�
#train/gradients/cost/Mean_grad/TileTile&train/gradients/cost/Mean_grad/Reshape$train/gradients/cost/Mean_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:���������
|
&train/gradients/cost/Mean_grad/Shape_1Shapecost/SquaredDifference*
T0*
out_type0*
_output_shapes
:
i
&train/gradients/cost/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
n
$train/gradients/cost/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
#train/gradients/cost/Mean_grad/ProdProd&train/gradients/cost/Mean_grad/Shape_1$train/gradients/cost/Mean_grad/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
p
&train/gradients/cost/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
%train/gradients/cost/Mean_grad/Prod_1Prod&train/gradients/cost/Mean_grad/Shape_2&train/gradients/cost/Mean_grad/Const_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
j
(train/gradients/cost/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
&train/gradients/cost/Mean_grad/MaximumMaximum%train/gradients/cost/Mean_grad/Prod_1(train/gradients/cost/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
�
'train/gradients/cost/Mean_grad/floordivFloorDiv#train/gradients/cost/Mean_grad/Prod&train/gradients/cost/Mean_grad/Maximum*
_output_shapes
: *
T0
�
#train/gradients/cost/Mean_grad/CastCast'train/gradients/cost/Mean_grad/floordiv*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0
�
&train/gradients/cost/Mean_grad/truedivRealDiv#train/gradients/cost/Mean_grad/Tile#train/gradients/cost/Mean_grad/Cast*
T0*'
_output_shapes
:���������
{
1train/gradients/cost/SquaredDifference_grad/ShapeShape
output/add*
T0*
out_type0*
_output_shapes
:
�
3train/gradients/cost/SquaredDifference_grad/Shape_1Shapecost/Placeholder*
T0*
out_type0*
_output_shapes
:
�
Atrain/gradients/cost/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgs1train/gradients/cost/SquaredDifference_grad/Shape3train/gradients/cost/SquaredDifference_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
2train/gradients/cost/SquaredDifference_grad/scalarConst'^train/gradients/cost/Mean_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
�
/train/gradients/cost/SquaredDifference_grad/mulMul2train/gradients/cost/SquaredDifference_grad/scalar&train/gradients/cost/Mean_grad/truediv*
T0*'
_output_shapes
:���������
�
/train/gradients/cost/SquaredDifference_grad/subSub
output/addcost/Placeholder'^train/gradients/cost/Mean_grad/truediv*
T0*'
_output_shapes
:���������
�
1train/gradients/cost/SquaredDifference_grad/mul_1Mul/train/gradients/cost/SquaredDifference_grad/mul/train/gradients/cost/SquaredDifference_grad/sub*
T0*'
_output_shapes
:���������
�
/train/gradients/cost/SquaredDifference_grad/SumSum1train/gradients/cost/SquaredDifference_grad/mul_1Atrain/gradients/cost/SquaredDifference_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
3train/gradients/cost/SquaredDifference_grad/ReshapeReshape/train/gradients/cost/SquaredDifference_grad/Sum1train/gradients/cost/SquaredDifference_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
1train/gradients/cost/SquaredDifference_grad/Sum_1Sum1train/gradients/cost/SquaredDifference_grad/mul_1Ctrain/gradients/cost/SquaredDifference_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
5train/gradients/cost/SquaredDifference_grad/Reshape_1Reshape1train/gradients/cost/SquaredDifference_grad/Sum_13train/gradients/cost/SquaredDifference_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
/train/gradients/cost/SquaredDifference_grad/NegNeg5train/gradients/cost/SquaredDifference_grad/Reshape_1*
T0*'
_output_shapes
:���������
�
<train/gradients/cost/SquaredDifference_grad/tuple/group_depsNoOp0^train/gradients/cost/SquaredDifference_grad/Neg4^train/gradients/cost/SquaredDifference_grad/Reshape
�
Dtrain/gradients/cost/SquaredDifference_grad/tuple/control_dependencyIdentity3train/gradients/cost/SquaredDifference_grad/Reshape=^train/gradients/cost/SquaredDifference_grad/tuple/group_deps*
T0*F
_class<
:8loc:@train/gradients/cost/SquaredDifference_grad/Reshape*'
_output_shapes
:���������
�
Ftrain/gradients/cost/SquaredDifference_grad/tuple/control_dependency_1Identity/train/gradients/cost/SquaredDifference_grad/Neg=^train/gradients/cost/SquaredDifference_grad/tuple/group_deps*
T0*B
_class8
64loc:@train/gradients/cost/SquaredDifference_grad/Neg*'
_output_shapes
:���������
r
%train/gradients/output/add_grad/ShapeShapeoutput/MatMul*
T0*
out_type0*
_output_shapes
:
q
'train/gradients/output/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
5train/gradients/output/add_grad/BroadcastGradientArgsBroadcastGradientArgs%train/gradients/output/add_grad/Shape'train/gradients/output/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
#train/gradients/output/add_grad/SumSumDtrain/gradients/cost/SquaredDifference_grad/tuple/control_dependency5train/gradients/output/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
'train/gradients/output/add_grad/ReshapeReshape#train/gradients/output/add_grad/Sum%train/gradients/output/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
%train/gradients/output/add_grad/Sum_1SumDtrain/gradients/cost/SquaredDifference_grad/tuple/control_dependency7train/gradients/output/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
)train/gradients/output/add_grad/Reshape_1Reshape%train/gradients/output/add_grad/Sum_1'train/gradients/output/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
�
0train/gradients/output/add_grad/tuple/group_depsNoOp(^train/gradients/output/add_grad/Reshape*^train/gradients/output/add_grad/Reshape_1
�
8train/gradients/output/add_grad/tuple/control_dependencyIdentity'train/gradients/output/add_grad/Reshape1^train/gradients/output/add_grad/tuple/group_deps*
T0*:
_class0
.,loc:@train/gradients/output/add_grad/Reshape*'
_output_shapes
:���������
�
:train/gradients/output/add_grad/tuple/control_dependency_1Identity)train/gradients/output/add_grad/Reshape_11^train/gradients/output/add_grad/tuple/group_deps*
T0*<
_class2
0.loc:@train/gradients/output/add_grad/Reshape_1*
_output_shapes
:
�
)train/gradients/output/MatMul_grad/MatMulMatMul8train/gradients/output/add_grad/tuple/control_dependencyoutput/weights4/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b(
�
+train/gradients/output/MatMul_grad/MatMul_1MatMullayer_3/Relu8train/gradients/output/add_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:*
transpose_a(
�
3train/gradients/output/MatMul_grad/tuple/group_depsNoOp*^train/gradients/output/MatMul_grad/MatMul,^train/gradients/output/MatMul_grad/MatMul_1
�
;train/gradients/output/MatMul_grad/tuple/control_dependencyIdentity)train/gradients/output/MatMul_grad/MatMul4^train/gradients/output/MatMul_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*<
_class2
0.loc:@train/gradients/output/MatMul_grad/MatMul
�
=train/gradients/output/MatMul_grad/tuple/control_dependency_1Identity+train/gradients/output/MatMul_grad/MatMul_14^train/gradients/output/MatMul_grad/tuple/group_deps*
T0*>
_class4
20loc:@train/gradients/output/MatMul_grad/MatMul_1*
_output_shapes

:
�
*train/gradients/layer_3/Relu_grad/ReluGradReluGrad;train/gradients/output/MatMul_grad/tuple/control_dependencylayer_3/Relu*
T0*'
_output_shapes
:���������
t
&train/gradients/layer_3/add_grad/ShapeShapelayer_3/MatMul*
_output_shapes
:*
T0*
out_type0
r
(train/gradients/layer_3/add_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
�
6train/gradients/layer_3/add_grad/BroadcastGradientArgsBroadcastGradientArgs&train/gradients/layer_3/add_grad/Shape(train/gradients/layer_3/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
$train/gradients/layer_3/add_grad/SumSum*train/gradients/layer_3/Relu_grad/ReluGrad6train/gradients/layer_3/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
(train/gradients/layer_3/add_grad/ReshapeReshape$train/gradients/layer_3/add_grad/Sum&train/gradients/layer_3/add_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
&train/gradients/layer_3/add_grad/Sum_1Sum*train/gradients/layer_3/Relu_grad/ReluGrad8train/gradients/layer_3/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
*train/gradients/layer_3/add_grad/Reshape_1Reshape&train/gradients/layer_3/add_grad/Sum_1(train/gradients/layer_3/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
�
1train/gradients/layer_3/add_grad/tuple/group_depsNoOp)^train/gradients/layer_3/add_grad/Reshape+^train/gradients/layer_3/add_grad/Reshape_1
�
9train/gradients/layer_3/add_grad/tuple/control_dependencyIdentity(train/gradients/layer_3/add_grad/Reshape2^train/gradients/layer_3/add_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*;
_class1
/-loc:@train/gradients/layer_3/add_grad/Reshape
�
;train/gradients/layer_3/add_grad/tuple/control_dependency_1Identity*train/gradients/layer_3/add_grad/Reshape_12^train/gradients/layer_3/add_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_3/add_grad/Reshape_1*
_output_shapes
:
�
*train/gradients/layer_3/MatMul_grad/MatMulMatMul9train/gradients/layer_3/add_grad/tuple/control_dependencylayer_3/weights3/read*
transpose_b(*
T0*'
_output_shapes
:���������*
transpose_a( 
�
,train/gradients/layer_3/MatMul_grad/MatMul_1MatMullayer_2/Relu9train/gradients/layer_3/add_grad/tuple/control_dependency*
T0*
_output_shapes

:*
transpose_a(*
transpose_b( 
�
4train/gradients/layer_3/MatMul_grad/tuple/group_depsNoOp+^train/gradients/layer_3/MatMul_grad/MatMul-^train/gradients/layer_3/MatMul_grad/MatMul_1
�
<train/gradients/layer_3/MatMul_grad/tuple/control_dependencyIdentity*train/gradients/layer_3/MatMul_grad/MatMul5^train/gradients/layer_3/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_3/MatMul_grad/MatMul*'
_output_shapes
:���������
�
>train/gradients/layer_3/MatMul_grad/tuple/control_dependency_1Identity,train/gradients/layer_3/MatMul_grad/MatMul_15^train/gradients/layer_3/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@train/gradients/layer_3/MatMul_grad/MatMul_1*
_output_shapes

:
�
*train/gradients/layer_2/Relu_grad/ReluGradReluGrad<train/gradients/layer_3/MatMul_grad/tuple/control_dependencylayer_2/Relu*'
_output_shapes
:���������*
T0
t
&train/gradients/layer_2/add_grad/ShapeShapelayer_2/MatMul*
T0*
out_type0*
_output_shapes
:
r
(train/gradients/layer_2/add_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
�
6train/gradients/layer_2/add_grad/BroadcastGradientArgsBroadcastGradientArgs&train/gradients/layer_2/add_grad/Shape(train/gradients/layer_2/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
$train/gradients/layer_2/add_grad/SumSum*train/gradients/layer_2/Relu_grad/ReluGrad6train/gradients/layer_2/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
(train/gradients/layer_2/add_grad/ReshapeReshape$train/gradients/layer_2/add_grad/Sum&train/gradients/layer_2/add_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
&train/gradients/layer_2/add_grad/Sum_1Sum*train/gradients/layer_2/Relu_grad/ReluGrad8train/gradients/layer_2/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
*train/gradients/layer_2/add_grad/Reshape_1Reshape&train/gradients/layer_2/add_grad/Sum_1(train/gradients/layer_2/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
�
1train/gradients/layer_2/add_grad/tuple/group_depsNoOp)^train/gradients/layer_2/add_grad/Reshape+^train/gradients/layer_2/add_grad/Reshape_1
�
9train/gradients/layer_2/add_grad/tuple/control_dependencyIdentity(train/gradients/layer_2/add_grad/Reshape2^train/gradients/layer_2/add_grad/tuple/group_deps*
T0*;
_class1
/-loc:@train/gradients/layer_2/add_grad/Reshape*'
_output_shapes
:���������
�
;train/gradients/layer_2/add_grad/tuple/control_dependency_1Identity*train/gradients/layer_2/add_grad/Reshape_12^train/gradients/layer_2/add_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_2/add_grad/Reshape_1*
_output_shapes
:
�
*train/gradients/layer_2/MatMul_grad/MatMulMatMul9train/gradients/layer_2/add_grad/tuple/control_dependencylayer_2/weights2/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b(
�
,train/gradients/layer_2/MatMul_grad/MatMul_1MatMullayer_1/Relu9train/gradients/layer_2/add_grad/tuple/control_dependency*
T0*
_output_shapes

:*
transpose_a(*
transpose_b( 
�
4train/gradients/layer_2/MatMul_grad/tuple/group_depsNoOp+^train/gradients/layer_2/MatMul_grad/MatMul-^train/gradients/layer_2/MatMul_grad/MatMul_1
�
<train/gradients/layer_2/MatMul_grad/tuple/control_dependencyIdentity*train/gradients/layer_2/MatMul_grad/MatMul5^train/gradients/layer_2/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_2/MatMul_grad/MatMul*'
_output_shapes
:���������
�
>train/gradients/layer_2/MatMul_grad/tuple/control_dependency_1Identity,train/gradients/layer_2/MatMul_grad/MatMul_15^train/gradients/layer_2/MatMul_grad/tuple/group_deps*
_output_shapes

:*
T0*?
_class5
31loc:@train/gradients/layer_2/MatMul_grad/MatMul_1
�
*train/gradients/layer_1/Relu_grad/ReluGradReluGrad<train/gradients/layer_2/MatMul_grad/tuple/control_dependencylayer_1/Relu*
T0*'
_output_shapes
:���������
t
&train/gradients/layer_1/add_grad/ShapeShapelayer_1/MatMul*
T0*
out_type0*
_output_shapes
:
r
(train/gradients/layer_1/add_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
�
6train/gradients/layer_1/add_grad/BroadcastGradientArgsBroadcastGradientArgs&train/gradients/layer_1/add_grad/Shape(train/gradients/layer_1/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
$train/gradients/layer_1/add_grad/SumSum*train/gradients/layer_1/Relu_grad/ReluGrad6train/gradients/layer_1/add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
(train/gradients/layer_1/add_grad/ReshapeReshape$train/gradients/layer_1/add_grad/Sum&train/gradients/layer_1/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
&train/gradients/layer_1/add_grad/Sum_1Sum*train/gradients/layer_1/Relu_grad/ReluGrad8train/gradients/layer_1/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
*train/gradients/layer_1/add_grad/Reshape_1Reshape&train/gradients/layer_1/add_grad/Sum_1(train/gradients/layer_1/add_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
�
1train/gradients/layer_1/add_grad/tuple/group_depsNoOp)^train/gradients/layer_1/add_grad/Reshape+^train/gradients/layer_1/add_grad/Reshape_1
�
9train/gradients/layer_1/add_grad/tuple/control_dependencyIdentity(train/gradients/layer_1/add_grad/Reshape2^train/gradients/layer_1/add_grad/tuple/group_deps*
T0*;
_class1
/-loc:@train/gradients/layer_1/add_grad/Reshape*'
_output_shapes
:���������
�
;train/gradients/layer_1/add_grad/tuple/control_dependency_1Identity*train/gradients/layer_1/add_grad/Reshape_12^train/gradients/layer_1/add_grad/tuple/group_deps*
_output_shapes
:*
T0*=
_class3
1/loc:@train/gradients/layer_1/add_grad/Reshape_1
�
*train/gradients/layer_1/MatMul_grad/MatMulMatMul9train/gradients/layer_1/add_grad/tuple/control_dependencylayer_1/weights1/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b(
�
,train/gradients/layer_1/MatMul_grad/MatMul_1MatMulinput/Placeholder9train/gradients/layer_1/add_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:*
transpose_a(
�
4train/gradients/layer_1/MatMul_grad/tuple/group_depsNoOp+^train/gradients/layer_1/MatMul_grad/MatMul-^train/gradients/layer_1/MatMul_grad/MatMul_1
�
<train/gradients/layer_1/MatMul_grad/tuple/control_dependencyIdentity*train/gradients/layer_1/MatMul_grad/MatMul5^train/gradients/layer_1/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_1/MatMul_grad/MatMul*'
_output_shapes
:���������
�
>train/gradients/layer_1/MatMul_grad/tuple/control_dependency_1Identity,train/gradients/layer_1/MatMul_grad/MatMul_15^train/gradients/layer_1/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@train/gradients/layer_1/MatMul_grad/MatMul_1*
_output_shapes

:
�
train/beta1_power/initial_valueConst*"
_class
loc:@layer_1/biases1*
valueB
 *fff?*
dtype0*
_output_shapes
: 
�
train/beta1_power
VariableV2*
dtype0*
_output_shapes
: *
shared_name *"
_class
loc:@layer_1/biases1*
	container *
shape: 
�
train/beta1_power/AssignAssigntrain/beta1_powertrain/beta1_power/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*"
_class
loc:@layer_1/biases1
z
train/beta1_power/readIdentitytrain/beta1_power*
T0*"
_class
loc:@layer_1/biases1*
_output_shapes
: 
�
train/beta2_power/initial_valueConst*"
_class
loc:@layer_1/biases1*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
train/beta2_power
VariableV2*
dtype0*
_output_shapes
: *
shared_name *"
_class
loc:@layer_1/biases1*
	container *
shape: 
�
train/beta2_power/AssignAssigntrain/beta2_powertrain/beta2_power/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*"
_class
loc:@layer_1/biases1
z
train/beta2_power/readIdentitytrain/beta2_power*
T0*"
_class
loc:@layer_1/biases1*
_output_shapes
: 
�
-train/layer_1/weights1/Adam/Initializer/zerosConst*#
_class
loc:@layer_1/weights1*
valueB*    *
dtype0*
_output_shapes

:
�
train/layer_1/weights1/Adam
VariableV2*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@layer_1/weights1*
	container *
shape
:
�
"train/layer_1/weights1/Adam/AssignAssigntrain/layer_1/weights1/Adam-train/layer_1/weights1/Adam/Initializer/zeros*
T0*#
_class
loc:@layer_1/weights1*
validate_shape(*
_output_shapes

:*
use_locking(
�
 train/layer_1/weights1/Adam/readIdentitytrain/layer_1/weights1/Adam*
T0*#
_class
loc:@layer_1/weights1*
_output_shapes

:
�
/train/layer_1/weights1/Adam_1/Initializer/zerosConst*#
_class
loc:@layer_1/weights1*
valueB*    *
dtype0*
_output_shapes

:
�
train/layer_1/weights1/Adam_1
VariableV2*
shared_name *#
_class
loc:@layer_1/weights1*
	container *
shape
:*
dtype0*
_output_shapes

:
�
$train/layer_1/weights1/Adam_1/AssignAssigntrain/layer_1/weights1/Adam_1/train/layer_1/weights1/Adam_1/Initializer/zeros*
T0*#
_class
loc:@layer_1/weights1*
validate_shape(*
_output_shapes

:*
use_locking(
�
"train/layer_1/weights1/Adam_1/readIdentitytrain/layer_1/weights1/Adam_1*
T0*#
_class
loc:@layer_1/weights1*
_output_shapes

:
�
,train/layer_1/biases1/Adam/Initializer/zerosConst*"
_class
loc:@layer_1/biases1*
valueB*    *
dtype0*
_output_shapes
:
�
train/layer_1/biases1/Adam
VariableV2*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@layer_1/biases1*
	container *
shape:
�
!train/layer_1/biases1/Adam/AssignAssigntrain/layer_1/biases1/Adam,train/layer_1/biases1/Adam/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
:
�
train/layer_1/biases1/Adam/readIdentitytrain/layer_1/biases1/Adam*
_output_shapes
:*
T0*"
_class
loc:@layer_1/biases1
�
.train/layer_1/biases1/Adam_1/Initializer/zerosConst*"
_class
loc:@layer_1/biases1*
valueB*    *
dtype0*
_output_shapes
:
�
train/layer_1/biases1/Adam_1
VariableV2*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@layer_1/biases1*
	container *
shape:
�
#train/layer_1/biases1/Adam_1/AssignAssigntrain/layer_1/biases1/Adam_1.train/layer_1/biases1/Adam_1/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
:
�
!train/layer_1/biases1/Adam_1/readIdentitytrain/layer_1/biases1/Adam_1*
T0*"
_class
loc:@layer_1/biases1*
_output_shapes
:
�
-train/layer_2/weights2/Adam/Initializer/zerosConst*#
_class
loc:@layer_2/weights2*
valueB*    *
dtype0*
_output_shapes

:
�
train/layer_2/weights2/Adam
VariableV2*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@layer_2/weights2*
	container *
shape
:
�
"train/layer_2/weights2/Adam/AssignAssigntrain/layer_2/weights2/Adam-train/layer_2/weights2/Adam/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@layer_2/weights2*
validate_shape(*
_output_shapes

:
�
 train/layer_2/weights2/Adam/readIdentitytrain/layer_2/weights2/Adam*
T0*#
_class
loc:@layer_2/weights2*
_output_shapes

:
�
/train/layer_2/weights2/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes

:*#
_class
loc:@layer_2/weights2*
valueB*    
�
train/layer_2/weights2/Adam_1
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@layer_2/weights2*
	container 
�
$train/layer_2/weights2/Adam_1/AssignAssigntrain/layer_2/weights2/Adam_1/train/layer_2/weights2/Adam_1/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@layer_2/weights2*
validate_shape(*
_output_shapes

:
�
"train/layer_2/weights2/Adam_1/readIdentitytrain/layer_2/weights2/Adam_1*
T0*#
_class
loc:@layer_2/weights2*
_output_shapes

:
�
,train/layer_2/biases2/Adam/Initializer/zerosConst*"
_class
loc:@layer_2/biases2*
valueB*    *
dtype0*
_output_shapes
:
�
train/layer_2/biases2/Adam
VariableV2*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@layer_2/biases2*
	container *
shape:
�
!train/layer_2/biases2/Adam/AssignAssigntrain/layer_2/biases2/Adam,train/layer_2/biases2/Adam/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@layer_2/biases2*
validate_shape(*
_output_shapes
:
�
train/layer_2/biases2/Adam/readIdentitytrain/layer_2/biases2/Adam*
_output_shapes
:*
T0*"
_class
loc:@layer_2/biases2
�
.train/layer_2/biases2/Adam_1/Initializer/zerosConst*"
_class
loc:@layer_2/biases2*
valueB*    *
dtype0*
_output_shapes
:
�
train/layer_2/biases2/Adam_1
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@layer_2/biases2*
	container 
�
#train/layer_2/biases2/Adam_1/AssignAssigntrain/layer_2/biases2/Adam_1.train/layer_2/biases2/Adam_1/Initializer/zeros*
T0*"
_class
loc:@layer_2/biases2*
validate_shape(*
_output_shapes
:*
use_locking(
�
!train/layer_2/biases2/Adam_1/readIdentitytrain/layer_2/biases2/Adam_1*
T0*"
_class
loc:@layer_2/biases2*
_output_shapes
:
�
-train/layer_3/weights3/Adam/Initializer/zerosConst*
dtype0*
_output_shapes

:*#
_class
loc:@layer_3/weights3*
valueB*    
�
train/layer_3/weights3/Adam
VariableV2*#
_class
loc:@layer_3/weights3*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
�
"train/layer_3/weights3/Adam/AssignAssigntrain/layer_3/weights3/Adam-train/layer_3/weights3/Adam/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@layer_3/weights3*
validate_shape(*
_output_shapes

:
�
 train/layer_3/weights3/Adam/readIdentitytrain/layer_3/weights3/Adam*
T0*#
_class
loc:@layer_3/weights3*
_output_shapes

:
�
/train/layer_3/weights3/Adam_1/Initializer/zerosConst*#
_class
loc:@layer_3/weights3*
valueB*    *
dtype0*
_output_shapes

:
�
train/layer_3/weights3/Adam_1
VariableV2*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@layer_3/weights3*
	container *
shape
:
�
$train/layer_3/weights3/Adam_1/AssignAssigntrain/layer_3/weights3/Adam_1/train/layer_3/weights3/Adam_1/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@layer_3/weights3*
validate_shape(*
_output_shapes

:
�
"train/layer_3/weights3/Adam_1/readIdentitytrain/layer_3/weights3/Adam_1*
T0*#
_class
loc:@layer_3/weights3*
_output_shapes

:
�
,train/layer_3/biases3/Adam/Initializer/zerosConst*"
_class
loc:@layer_3/biases3*
valueB*    *
dtype0*
_output_shapes
:
�
train/layer_3/biases3/Adam
VariableV2*"
_class
loc:@layer_3/biases3*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
�
!train/layer_3/biases3/Adam/AssignAssigntrain/layer_3/biases3/Adam,train/layer_3/biases3/Adam/Initializer/zeros*
T0*"
_class
loc:@layer_3/biases3*
validate_shape(*
_output_shapes
:*
use_locking(
�
train/layer_3/biases3/Adam/readIdentitytrain/layer_3/biases3/Adam*
_output_shapes
:*
T0*"
_class
loc:@layer_3/biases3
�
.train/layer_3/biases3/Adam_1/Initializer/zerosConst*"
_class
loc:@layer_3/biases3*
valueB*    *
dtype0*
_output_shapes
:
�
train/layer_3/biases3/Adam_1
VariableV2*
shared_name *"
_class
loc:@layer_3/biases3*
	container *
shape:*
dtype0*
_output_shapes
:
�
#train/layer_3/biases3/Adam_1/AssignAssigntrain/layer_3/biases3/Adam_1.train/layer_3/biases3/Adam_1/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@layer_3/biases3*
validate_shape(*
_output_shapes
:
�
!train/layer_3/biases3/Adam_1/readIdentitytrain/layer_3/biases3/Adam_1*
T0*"
_class
loc:@layer_3/biases3*
_output_shapes
:
�
,train/output/weights4/Adam/Initializer/zerosConst*"
_class
loc:@output/weights4*
valueB*    *
dtype0*
_output_shapes

:
�
train/output/weights4/Adam
VariableV2*"
_class
loc:@output/weights4*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
�
!train/output/weights4/Adam/AssignAssigntrain/output/weights4/Adam,train/output/weights4/Adam/Initializer/zeros*
T0*"
_class
loc:@output/weights4*
validate_shape(*
_output_shapes

:*
use_locking(
�
train/output/weights4/Adam/readIdentitytrain/output/weights4/Adam*
T0*"
_class
loc:@output/weights4*
_output_shapes

:
�
.train/output/weights4/Adam_1/Initializer/zerosConst*"
_class
loc:@output/weights4*
valueB*    *
dtype0*
_output_shapes

:
�
train/output/weights4/Adam_1
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *"
_class
loc:@output/weights4*
	container 
�
#train/output/weights4/Adam_1/AssignAssigntrain/output/weights4/Adam_1.train/output/weights4/Adam_1/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@output/weights4*
validate_shape(*
_output_shapes

:
�
!train/output/weights4/Adam_1/readIdentitytrain/output/weights4/Adam_1*
_output_shapes

:*
T0*"
_class
loc:@output/weights4
�
+train/output/biases4/Adam/Initializer/zerosConst*!
_class
loc:@output/biases4*
valueB*    *
dtype0*
_output_shapes
:
�
train/output/biases4/Adam
VariableV2*!
_class
loc:@output/biases4*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
�
 train/output/biases4/Adam/AssignAssigntrain/output/biases4/Adam+train/output/biases4/Adam/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@output/biases4*
validate_shape(*
_output_shapes
:
�
train/output/biases4/Adam/readIdentitytrain/output/biases4/Adam*
T0*!
_class
loc:@output/biases4*
_output_shapes
:
�
-train/output/biases4/Adam_1/Initializer/zerosConst*!
_class
loc:@output/biases4*
valueB*    *
dtype0*
_output_shapes
:
�
train/output/biases4/Adam_1
VariableV2*
dtype0*
_output_shapes
:*
shared_name *!
_class
loc:@output/biases4*
	container *
shape:
�
"train/output/biases4/Adam_1/AssignAssigntrain/output/biases4/Adam_1-train/output/biases4/Adam_1/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@output/biases4*
validate_shape(*
_output_shapes
:
�
 train/output/biases4/Adam_1/readIdentitytrain/output/biases4/Adam_1*
_output_shapes
:*
T0*!
_class
loc:@output/biases4
]
train/Adam/learning_rateConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
U
train/Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
U
train/Adam/beta2Const*
valueB
 *w�?*
dtype0*
_output_shapes
: 
W
train/Adam/epsilonConst*
valueB
 *w�+2*
dtype0*
_output_shapes
: 
�
,train/Adam/update_layer_1/weights1/ApplyAdam	ApplyAdamlayer_1/weights1train/layer_1/weights1/Adamtrain/layer_1/weights1/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon>train/gradients/layer_1/MatMul_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes

:*
use_locking( *
T0*#
_class
loc:@layer_1/weights1
�
+train/Adam/update_layer_1/biases1/ApplyAdam	ApplyAdamlayer_1/biases1train/layer_1/biases1/Adamtrain/layer_1/biases1/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon;train/gradients/layer_1/add_grad/tuple/control_dependency_1*
T0*"
_class
loc:@layer_1/biases1*
use_nesterov( *
_output_shapes
:*
use_locking( 
�
,train/Adam/update_layer_2/weights2/ApplyAdam	ApplyAdamlayer_2/weights2train/layer_2/weights2/Adamtrain/layer_2/weights2/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon>train/gradients/layer_2/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*#
_class
loc:@layer_2/weights2*
use_nesterov( *
_output_shapes

:
�
+train/Adam/update_layer_2/biases2/ApplyAdam	ApplyAdamlayer_2/biases2train/layer_2/biases2/Adamtrain/layer_2/biases2/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon;train/gradients/layer_2/add_grad/tuple/control_dependency_1*
use_locking( *
T0*"
_class
loc:@layer_2/biases2*
use_nesterov( *
_output_shapes
:
�
,train/Adam/update_layer_3/weights3/ApplyAdam	ApplyAdamlayer_3/weights3train/layer_3/weights3/Adamtrain/layer_3/weights3/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon>train/gradients/layer_3/MatMul_grad/tuple/control_dependency_1*
T0*#
_class
loc:@layer_3/weights3*
use_nesterov( *
_output_shapes

:*
use_locking( 
�
+train/Adam/update_layer_3/biases3/ApplyAdam	ApplyAdamlayer_3/biases3train/layer_3/biases3/Adamtrain/layer_3/biases3/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon;train/gradients/layer_3/add_grad/tuple/control_dependency_1*
use_locking( *
T0*"
_class
loc:@layer_3/biases3*
use_nesterov( *
_output_shapes
:
�
+train/Adam/update_output/weights4/ApplyAdam	ApplyAdamoutput/weights4train/output/weights4/Adamtrain/output/weights4/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon=train/gradients/output/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*"
_class
loc:@output/weights4*
use_nesterov( *
_output_shapes

:
�
*train/Adam/update_output/biases4/ApplyAdam	ApplyAdamoutput/biases4train/output/biases4/Adamtrain/output/biases4/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon:train/gradients/output/add_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0*!
_class
loc:@output/biases4
�
train/Adam/mulMultrain/beta1_power/readtrain/Adam/beta1,^train/Adam/update_layer_1/biases1/ApplyAdam-^train/Adam/update_layer_1/weights1/ApplyAdam,^train/Adam/update_layer_2/biases2/ApplyAdam-^train/Adam/update_layer_2/weights2/ApplyAdam,^train/Adam/update_layer_3/biases3/ApplyAdam-^train/Adam/update_layer_3/weights3/ApplyAdam+^train/Adam/update_output/biases4/ApplyAdam,^train/Adam/update_output/weights4/ApplyAdam*
T0*"
_class
loc:@layer_1/biases1*
_output_shapes
: 
�
train/Adam/AssignAssigntrain/beta1_powertrain/Adam/mul*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
: *
use_locking( 
�
train/Adam/mul_1Multrain/beta2_power/readtrain/Adam/beta2,^train/Adam/update_layer_1/biases1/ApplyAdam-^train/Adam/update_layer_1/weights1/ApplyAdam,^train/Adam/update_layer_2/biases2/ApplyAdam-^train/Adam/update_layer_2/weights2/ApplyAdam,^train/Adam/update_layer_3/biases3/ApplyAdam-^train/Adam/update_layer_3/weights3/ApplyAdam+^train/Adam/update_output/biases4/ApplyAdam,^train/Adam/update_output/weights4/ApplyAdam*
T0*"
_class
loc:@layer_1/biases1*
_output_shapes
: 
�
train/Adam/Assign_1Assigntrain/beta2_powertrain/Adam/mul_1*
use_locking( *
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
: 
�

train/AdamNoOp^train/Adam/Assign^train/Adam/Assign_1,^train/Adam/update_layer_1/biases1/ApplyAdam-^train/Adam/update_layer_1/weights1/ApplyAdam,^train/Adam/update_layer_2/biases2/ApplyAdam-^train/Adam/update_layer_2/weights2/ApplyAdam,^train/Adam/update_layer_3/biases3/ApplyAdam-^train/Adam/update_layer_3/weights3/ApplyAdam+^train/Adam/update_output/biases4/ApplyAdam,^train/Adam/update_output/weights4/ApplyAdam
n
logging/current_cost/tagsConst*
dtype0*
_output_shapes
: *%
valueB Blogging/current_cost
l
logging/current_costScalarSummarylogging/current_cost/tags	cost/Mean*
T0*
_output_shapes
: 
a
logging/Merge/MergeSummaryMergeSummarylogging/current_cost*
N*
_output_shapes
: 
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
�
save/SaveV2/tensor_namesConst*
dtype0*
_output_shapes
:*�
value�B�Blayer_1/biases1Blayer_1/weights1Blayer_2/biases2Blayer_2/weights2Blayer_3/biases3Blayer_3/weights3Boutput/biases4Boutput/weights4Btrain/beta1_powerBtrain/beta2_powerBtrain/layer_1/biases1/AdamBtrain/layer_1/biases1/Adam_1Btrain/layer_1/weights1/AdamBtrain/layer_1/weights1/Adam_1Btrain/layer_2/biases2/AdamBtrain/layer_2/biases2/Adam_1Btrain/layer_2/weights2/AdamBtrain/layer_2/weights2/Adam_1Btrain/layer_3/biases3/AdamBtrain/layer_3/biases3/Adam_1Btrain/layer_3/weights3/AdamBtrain/layer_3/weights3/Adam_1Btrain/output/biases4/AdamBtrain/output/biases4/Adam_1Btrain/output/weights4/AdamBtrain/output/weights4/Adam_1
�
save/SaveV2/shape_and_slicesConst*
dtype0*
_output_shapes
:*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B 
�
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_sliceslayer_1/biases1layer_1/weights1layer_2/biases2layer_2/weights2layer_3/biases3layer_3/weights3output/biases4output/weights4train/beta1_powertrain/beta2_powertrain/layer_1/biases1/Adamtrain/layer_1/biases1/Adam_1train/layer_1/weights1/Adamtrain/layer_1/weights1/Adam_1train/layer_2/biases2/Adamtrain/layer_2/biases2/Adam_1train/layer_2/weights2/Adamtrain/layer_2/weights2/Adam_1train/layer_3/biases3/Adamtrain/layer_3/biases3/Adam_1train/layer_3/weights3/Adamtrain/layer_3/weights3/Adam_1train/output/biases4/Adamtrain/output/biases4/Adam_1train/output/weights4/Adamtrain/output/weights4/Adam_1*(
dtypes
2
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const*
_output_shapes
: 
�
save/RestoreV2/tensor_namesConst"/device:CPU:0*�
value�B�Blayer_1/biases1Blayer_1/weights1Blayer_2/biases2Blayer_2/weights2Blayer_3/biases3Blayer_3/weights3Boutput/biases4Boutput/weights4Btrain/beta1_powerBtrain/beta2_powerBtrain/layer_1/biases1/AdamBtrain/layer_1/biases1/Adam_1Btrain/layer_1/weights1/AdamBtrain/layer_1/weights1/Adam_1Btrain/layer_2/biases2/AdamBtrain/layer_2/biases2/Adam_1Btrain/layer_2/weights2/AdamBtrain/layer_2/weights2/Adam_1Btrain/layer_3/biases3/AdamBtrain/layer_3/biases3/Adam_1Btrain/layer_3/weights3/AdamBtrain/layer_3/weights3/Adam_1Btrain/output/biases4/AdamBtrain/output/biases4/Adam_1Btrain/output/weights4/AdamBtrain/output/weights4/Adam_1*
dtype0*
_output_shapes
:
�
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
�
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*|
_output_shapesj
h::::::::::::::::::::::::::*(
dtypes
2
�
save/AssignAssignlayer_1/biases1save/RestoreV2*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
:*
use_locking(
�
save/Assign_1Assignlayer_1/weights1save/RestoreV2:1*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@layer_1/weights1
�
save/Assign_2Assignlayer_2/biases2save/RestoreV2:2*
T0*"
_class
loc:@layer_2/biases2*
validate_shape(*
_output_shapes
:*
use_locking(
�
save/Assign_3Assignlayer_2/weights2save/RestoreV2:3*
T0*#
_class
loc:@layer_2/weights2*
validate_shape(*
_output_shapes

:*
use_locking(
�
save/Assign_4Assignlayer_3/biases3save/RestoreV2:4*
use_locking(*
T0*"
_class
loc:@layer_3/biases3*
validate_shape(*
_output_shapes
:
�
save/Assign_5Assignlayer_3/weights3save/RestoreV2:5*
use_locking(*
T0*#
_class
loc:@layer_3/weights3*
validate_shape(*
_output_shapes

:
�
save/Assign_6Assignoutput/biases4save/RestoreV2:6*
use_locking(*
T0*!
_class
loc:@output/biases4*
validate_shape(*
_output_shapes
:
�
save/Assign_7Assignoutput/weights4save/RestoreV2:7*
T0*"
_class
loc:@output/weights4*
validate_shape(*
_output_shapes

:*
use_locking(
�
save/Assign_8Assigntrain/beta1_powersave/RestoreV2:8*
use_locking(*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
: 
�
save/Assign_9Assigntrain/beta2_powersave/RestoreV2:9*
use_locking(*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
: 
�
save/Assign_10Assigntrain/layer_1/biases1/Adamsave/RestoreV2:10*
use_locking(*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
:
�
save/Assign_11Assigntrain/layer_1/biases1/Adam_1save/RestoreV2:11*
use_locking(*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
:
�
save/Assign_12Assigntrain/layer_1/weights1/Adamsave/RestoreV2:12*
use_locking(*
T0*#
_class
loc:@layer_1/weights1*
validate_shape(*
_output_shapes

:
�
save/Assign_13Assigntrain/layer_1/weights1/Adam_1save/RestoreV2:13*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@layer_1/weights1
�
save/Assign_14Assigntrain/layer_2/biases2/Adamsave/RestoreV2:14*
T0*"
_class
loc:@layer_2/biases2*
validate_shape(*
_output_shapes
:*
use_locking(
�
save/Assign_15Assigntrain/layer_2/biases2/Adam_1save/RestoreV2:15*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@layer_2/biases2
�
save/Assign_16Assigntrain/layer_2/weights2/Adamsave/RestoreV2:16*
T0*#
_class
loc:@layer_2/weights2*
validate_shape(*
_output_shapes

:*
use_locking(
�
save/Assign_17Assigntrain/layer_2/weights2/Adam_1save/RestoreV2:17*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@layer_2/weights2
�
save/Assign_18Assigntrain/layer_3/biases3/Adamsave/RestoreV2:18*
use_locking(*
T0*"
_class
loc:@layer_3/biases3*
validate_shape(*
_output_shapes
:
�
save/Assign_19Assigntrain/layer_3/biases3/Adam_1save/RestoreV2:19*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@layer_3/biases3
�
save/Assign_20Assigntrain/layer_3/weights3/Adamsave/RestoreV2:20*
use_locking(*
T0*#
_class
loc:@layer_3/weights3*
validate_shape(*
_output_shapes

:
�
save/Assign_21Assigntrain/layer_3/weights3/Adam_1save/RestoreV2:21*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@layer_3/weights3
�
save/Assign_22Assigntrain/output/biases4/Adamsave/RestoreV2:22*
use_locking(*
T0*!
_class
loc:@output/biases4*
validate_shape(*
_output_shapes
:
�
save/Assign_23Assigntrain/output/biases4/Adam_1save/RestoreV2:23*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*!
_class
loc:@output/biases4
�
save/Assign_24Assigntrain/output/weights4/Adamsave/RestoreV2:24*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*"
_class
loc:@output/weights4
�
save/Assign_25Assigntrain/output/weights4/Adam_1save/RestoreV2:25*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*"
_class
loc:@output/weights4
�
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
�
initNoOp^layer_1/biases1/Assign^layer_1/weights1/Assign^layer_2/biases2/Assign^layer_2/weights2/Assign^layer_3/biases3/Assign^layer_3/weights3/Assign^output/biases4/Assign^output/weights4/Assign^train/beta1_power/Assign^train/beta2_power/Assign"^train/layer_1/biases1/Adam/Assign$^train/layer_1/biases1/Adam_1/Assign#^train/layer_1/weights1/Adam/Assign%^train/layer_1/weights1/Adam_1/Assign"^train/layer_2/biases2/Adam/Assign$^train/layer_2/biases2/Adam_1/Assign#^train/layer_2/weights2/Adam/Assign%^train/layer_2/weights2/Adam_1/Assign"^train/layer_3/biases3/Adam/Assign$^train/layer_3/biases3/Adam_1/Assign#^train/layer_3/weights3/Adam/Assign%^train/layer_3/weights3/Adam_1/Assign!^train/output/biases4/Adam/Assign#^train/output/biases4/Adam_1/Assign"^train/output/weights4/Adam/Assign$^train/output/weights4/Adam_1/Assign""�
trainable_variables��
w
layer_1/weights1:0layer_1/weights1/Assignlayer_1/weights1/read:02-layer_1/weights1/Initializer/random_uniform:08
j
layer_1/biases1:0layer_1/biases1/Assignlayer_1/biases1/read:02#layer_1/biases1/Initializer/zeros:08
w
layer_2/weights2:0layer_2/weights2/Assignlayer_2/weights2/read:02-layer_2/weights2/Initializer/random_uniform:08
j
layer_2/biases2:0layer_2/biases2/Assignlayer_2/biases2/read:02#layer_2/biases2/Initializer/zeros:08
w
layer_3/weights3:0layer_3/weights3/Assignlayer_3/weights3/read:02-layer_3/weights3/Initializer/random_uniform:08
j
layer_3/biases3:0layer_3/biases3/Assignlayer_3/biases3/read:02#layer_3/biases3/Initializer/zeros:08
s
output/weights4:0output/weights4/Assignoutput/weights4/read:02,output/weights4/Initializer/random_uniform:08
f
output/biases4:0output/biases4/Assignoutput/biases4/read:02"output/biases4/Initializer/zeros:08"'
	summaries

logging/current_cost:0"
train_op


train/Adam"�
	variables��
w
layer_1/weights1:0layer_1/weights1/Assignlayer_1/weights1/read:02-layer_1/weights1/Initializer/random_uniform:08
j
layer_1/biases1:0layer_1/biases1/Assignlayer_1/biases1/read:02#layer_1/biases1/Initializer/zeros:08
w
layer_2/weights2:0layer_2/weights2/Assignlayer_2/weights2/read:02-layer_2/weights2/Initializer/random_uniform:08
j
layer_2/biases2:0layer_2/biases2/Assignlayer_2/biases2/read:02#layer_2/biases2/Initializer/zeros:08
w
layer_3/weights3:0layer_3/weights3/Assignlayer_3/weights3/read:02-layer_3/weights3/Initializer/random_uniform:08
j
layer_3/biases3:0layer_3/biases3/Assignlayer_3/biases3/read:02#layer_3/biases3/Initializer/zeros:08
s
output/weights4:0output/weights4/Assignoutput/weights4/read:02,output/weights4/Initializer/random_uniform:08
f
output/biases4:0output/biases4/Assignoutput/biases4/read:02"output/biases4/Initializer/zeros:08
l
train/beta1_power:0train/beta1_power/Assigntrain/beta1_power/read:02!train/beta1_power/initial_value:0
l
train/beta2_power:0train/beta2_power/Assigntrain/beta2_power/read:02!train/beta2_power/initial_value:0
�
train/layer_1/weights1/Adam:0"train/layer_1/weights1/Adam/Assign"train/layer_1/weights1/Adam/read:02/train/layer_1/weights1/Adam/Initializer/zeros:0
�
train/layer_1/weights1/Adam_1:0$train/layer_1/weights1/Adam_1/Assign$train/layer_1/weights1/Adam_1/read:021train/layer_1/weights1/Adam_1/Initializer/zeros:0
�
train/layer_1/biases1/Adam:0!train/layer_1/biases1/Adam/Assign!train/layer_1/biases1/Adam/read:02.train/layer_1/biases1/Adam/Initializer/zeros:0
�
train/layer_1/biases1/Adam_1:0#train/layer_1/biases1/Adam_1/Assign#train/layer_1/biases1/Adam_1/read:020train/layer_1/biases1/Adam_1/Initializer/zeros:0
�
train/layer_2/weights2/Adam:0"train/layer_2/weights2/Adam/Assign"train/layer_2/weights2/Adam/read:02/train/layer_2/weights2/Adam/Initializer/zeros:0
�
train/layer_2/weights2/Adam_1:0$train/layer_2/weights2/Adam_1/Assign$train/layer_2/weights2/Adam_1/read:021train/layer_2/weights2/Adam_1/Initializer/zeros:0
�
train/layer_2/biases2/Adam:0!train/layer_2/biases2/Adam/Assign!train/layer_2/biases2/Adam/read:02.train/layer_2/biases2/Adam/Initializer/zeros:0
�
train/layer_2/biases2/Adam_1:0#train/layer_2/biases2/Adam_1/Assign#train/layer_2/biases2/Adam_1/read:020train/layer_2/biases2/Adam_1/Initializer/zeros:0
�
train/layer_3/weights3/Adam:0"train/layer_3/weights3/Adam/Assign"train/layer_3/weights3/Adam/read:02/train/layer_3/weights3/Adam/Initializer/zeros:0
�
train/layer_3/weights3/Adam_1:0$train/layer_3/weights3/Adam_1/Assign$train/layer_3/weights3/Adam_1/read:021train/layer_3/weights3/Adam_1/Initializer/zeros:0
�
train/layer_3/biases3/Adam:0!train/layer_3/biases3/Adam/Assign!train/layer_3/biases3/Adam/read:02.train/layer_3/biases3/Adam/Initializer/zeros:0
�
train/layer_3/biases3/Adam_1:0#train/layer_3/biases3/Adam_1/Assign#train/layer_3/biases3/Adam_1/read:020train/layer_3/biases3/Adam_1/Initializer/zeros:0
�
train/output/weights4/Adam:0!train/output/weights4/Adam/Assign!train/output/weights4/Adam/read:02.train/output/weights4/Adam/Initializer/zeros:0
�
train/output/weights4/Adam_1:0#train/output/weights4/Adam_1/Assign#train/output/weights4/Adam_1/read:020train/output/weights4/Adam_1/Initializer/zeros:0
�
train/output/biases4/Adam:0 train/output/biases4/Adam/Assign train/output/biases4/Adam/read:02-train/output/biases4/Adam/Initializer/zeros:0
�
train/output/biases4/Adam_1:0"train/output/biases4/Adam_1/Assign"train/output/biases4/Adam_1/read:02/train/output/biases4/Adam_1/Initializer/zeros:0)Վ�(       �pJ	��2��A*

logging/current_cost2\>�Z�*       ����	�33��A*

logging/current_cost '�=�nE*       ����	Vc3��A
*

logging/current_cost¹�=�~=*       ����	��3��A*

logging/current_costg0�=9�δ*       ����	��3��A*

logging/current_cost���=�Z�;*       ����	��3��A*

logging/current_cost�И=�rsj*       ����	#4��A*

logging/current_cost�-�=)��*       ����	MQ4��A#*

logging/current_costP݀=XG��*       ����	��4��A(*

logging/current_cost.xm=盚�*       ����	|�4��A-*

logging/current_cost�I]=�Y}�*       ����	O�4��A2*

logging/current_cost��O=�o�*       ����	�5��A7*

logging/current_cost�D=�sv*       ����	=95��A<*

logging/current_cost�9<=�`�*       ����	h5��AA*

logging/current_cost��4=����*       ����	ԓ5��AF*

logging/current_cost0�-=��v*       ����	s�5��AK*

logging/current_costj(=�y�*       ����	��5��AP*

logging/current_cost�4#=3��#*       ����	�6��AU*

logging/current_cost`Y=��M�*       ����	�J6��AZ*

logging/current_cost��=�1*       ����	-z6��A_*

logging/current_cost�=����*       ����	��6��Ad*

logging/current_cost=�=����*       ����	�6��Ai*

logging/current_cost��=ۧ�*       ����	#7��An*

logging/current_cost�	=�u��*       ����	S.7��As*

logging/current_cost$r	=Cu</*       ����	zZ7��Ax*

logging/current_costk�=���{*       ����	�7��A}*

logging/current_cost�=�A
�+       ��K	�7��A�*

logging/current_cost �=��
�+       ��K	~�7��A�*

logging/current_cost�� ='��O+       ��K	�8��A�*

logging/current_cost�b�<�c%+       ��K	@8��A�*

logging/current_cost9��<��T�+       ��K	�m8��A�*

logging/current_cost��<��2	+       ��K	�8��A�*

logging/current_coste,�<�V/�+       ��K	C�8��A�*

logging/current_cost��<-�3_+       ��K	i�8��A�*

logging/current_cost�p�<r��y+       ��K	09��A�*

logging/current_cost�"�<%��+       ��K	Ab9��A�*

logging/current_cost��<|o�+       ��K	��9��A�*

logging/current_cost���<'�+       ��K	��9��A�*

logging/current_costG��<kz�j+       ��K	��9��A�*

logging/current_costD �<�x�V+       ��K	
':��A�*

logging/current_costk)�<߫U+       ��K	�Z:��A�*

logging/current_cost��<j�7+       ��K	�:��A�*

logging/current_cost���<��+       ��K	x�:��A�*

logging/current_cost��<H[��+       ��K	��:��A�*

logging/current_cost��<W쿎+       ��K	�;��A�*

logging/current_cost ��<��~+       ��K	jJ;��A�*

logging/current_costO�<�Se+       ��K	�;��A�*

logging/current_cost��<
S�+       ��K	��;��A�*

logging/current_cost��<�Y^�+       ��K	�<��A�*

logging/current_costN�<�{U?+       ��K	�X<��A�*

logging/current_costl�</�+       ��K	��<��A�*

logging/current_costtF�<+�yb+       ��K	4�<��A�*

logging/current_cost�m�<ń�+       ��K	�=��A�*

logging/current_cost���<����+       ��K		X=��A�*

logging/current_cost���<�� +       ��K	~�=��A�*

logging/current_cost̶�<�JUk+       ��K	��=��A�*

logging/current_cost�ˆ<s��V+       ��K	�>��A�*

logging/current_cost��<F��+       ��K	O>��A�*

logging/current_cost�{<©|+       ��K	�>��A�*

logging/current_cost��q<���V+       ��K	,�>��A�*

logging/current_cost7.i<`��+       ��K	;�>��A�*

logging/current_cost�6a<����+       ��K	[ ?��A�*

logging/current_cost��X<�'�+       ��K	�l?��A�*

logging/current_cost�Q<s�Է+       ��K	.�?��A�*

logging/current_costɡH<�,G�+       ��K	(�?��A�*

logging/current_cost�A<2/+       ��K	�@��A�*

logging/current_cost�;<tH�++       ��K	�;@��A�*

logging/current_costH5<m��+       ��K	�q@��A�*

logging/current_cost�d0<��+       ��K	��@��A�*

logging/current_cost��+<�y�U+       ��K	�@��A�*

logging/current_costT(<���K+       ��K	�A��A�*

logging/current_costf%<v�~B+       ��K	Q?A��A�*

logging/current_cost��"<��+       ��K	�pA��A�*

logging/current_cost� <���+       ��K	�A��A�*

logging/current_cost�[<x�}}+       ��K	��A��A�*

logging/current_cost�
<o�V+       ��K	sB��A�*

logging/current_costr<��6+       ��K	�SB��A�*

logging/current_cost�\<��j�+       ��K	M�B��A�*

logging/current_cost��<xa�%+       ��K	��B��A�*

logging/current_coste<*�/+       ��K	��B��A�*

logging/current_cost �<(�4�+       ��K	p/C��A�*

logging/current_cost�_<?��	+       ��K	�aC��A�*

logging/current_cost�A<����+       ��K	ҔC��A�*

logging/current_cost�#<�n +       ��K	��C��A�*

logging/current_costb<*	+       ��K	R�C��A�*

logging/current_cost�&<��g�+       ��K	q*D��A�*

logging/current_cost�W<�D��+       ��K	leD��A�*

logging/current_cost��<���]+       ��K	ڗD��A�*

logging/current_cost4�
<?Bg�+       ��K	}�D��A�*

logging/current_cost�(
<�L��+       ��K	E��A�*

logging/current_cost�	<���+       ��K	~1E��A�*

logging/current_cost��<�Dqw+       ��K		dE��A�*

logging/current_costra<z�+       ��K	��E��A�*

logging/current_cost��<㱲�+       ��K	x�E��A�*

logging/current_costu^<��+       ��K	�F��A�*

logging/current_cost��<��Z_+       ��K	4F��A�*

logging/current_cost��<ݛ ^+       ��K	�gF��A�*

logging/current_cost>!<�A�+       ��K	J�F��A�*

logging/current_cost��<e�+       ��K	��F��A�*

logging/current_cost�e<5F�h+       ��K	<G��A�*

logging/current_cost5<5d�+       ��K	��G��A�*

logging/current_cost��<$�\�+       ��K	��G��A�*

logging/current_cost�g<H^��+       ��K	�
H��A�*

logging/current_costd<�*�+       ��K	�LH��A�*

logging/current_cost��<���M+       ��K	ЍH��A�*

logging/current_cost]�<k���+       ��K	��H��A�*

logging/current_cost�S<�e+       ��K	)I��A�*

logging/current_cost�<�F�.+       ��K	~EI��A�*

logging/current_cost��<�Bm]+       ��K	T�I��A�*

logging/current_cost��<_�k+       ��K	:�I��A�*

logging/current_costI<L�K +       ��K	��I��A�*

logging/current_cost�<<�X+       ��K	�!J��A�*

logging/current_cost�<��t+       ��K	�VJ��A�*

logging/current_costߩ<f�A�+       ��K	։J��A�*

logging/current_costN{<�ȡ+       ��K	��J��A�*

logging/current_cost�G<�W +       ��K	��J��A�*

logging/current_cost�<xg�D+       ��K	'$K��A�*

logging/current_cost`� <�/�S+       ��K	�YK��A�*

logging/current_cost� <�M+       ��K	��K��A�*

logging/current_cost2e <#8yG+       ��K	.�K��A�*

logging/current_costz, <\C+       ��K	��K��A�*

logging/current_cost��; �\M+       ��K	�*L��A�*

logging/current_costܒ�;X�Z"+       ��K	[L��A�*

logging/current_cost�;�;U}RH+       ��K	čL��A�*

logging/current_cost��;X�߇+       ��K	�L��A�*

logging/current_cost[��;��yO+       ��K	��L��A�*

logging/current_cost�;�;H +       ��K	@;M��A�*

logging/current_costk��;�ڡ+       ��K	�M��A�*

logging/current_cost��;�e��+       ��K	P�M��A�*

logging/current_cost�Q�;�]a�+       ��K	�M��A�*

logging/current_costp
�;]�F+       ��K	/N��A�*

logging/current_cost��;ĥ]�+       ��K	�ON��A�*

logging/current_costr��;Զ�i+       ��K	D�N��A�*

logging/current_cost�8�;u�Bg+       ��K	��N��A�*

logging/current_cost  �;�{+       ��K	��N��A�*

logging/current_cost[��;�\�G+       ��K	�O��A�*

logging/current_costp��;0��M+       ��K	>[O��A�*

logging/current_cost��;I���+       ��K	�O��A�*

logging/current_cost;A�;U��+       ��K	��O��A�*

logging/current_cost�%�;3�r+       ��K	��O��A�*

logging/current_cost��;�x�@+       ��K	D-P��A�*

logging/current_cost��;Z}�t+       ��K	�bP��A�*

logging/current_cost���;����+       ��K	S�P��A�*

logging/current_cost.��;Uf�+       ��K	��P��A�*

logging/current_cost�l�;%�=�+       ��K	��P��A�*

logging/current_cost�J�;%߽u+       ��K	'Q��A�*

logging/current_costr*�;Y+       ��K	�WQ��A�*

logging/current_cost��;5
G�+       ��K	W�Q��A�*

logging/current_cost �;�4��+       ��K	��Q��A�*

logging/current_cost~�;d���+       ��K	��Q��A�*

logging/current_cost���;��K+       ��K	3(R��A�*

logging/current_cost���;o�qe+       ��K	�YR��A�*

logging/current_cost ��;5�j+       ��K	�R��A�*

logging/current_cost���;L�+       ��K	|�R��A�*

logging/current_costdv�;�k{P+       ��K	r�R��A�*

logging/current_cost�O�;f睛+       ��K	�'S��A�*

logging/current_cost�2�;셴+       ��K	�]S��A�*

logging/current_cost�;�=��+       ��K	0�S��A�*

logging/current_cost��;����+       ��K	��S��A�*

logging/current_cost'��;�-+       ��K	�
T��A�*

logging/current_cost���;lx�M+       ��K	sAT��A�*

logging/current_cost��;Cȹ�+       ��K	�rT��A�*

logging/current_cost��;@"dz+       ��K	��T��A�*

logging/current_costK��;�tƁ+       ��K	��T��A�*

logging/current_cost�r�;�i}\+       ��K	U��A�*

logging/current_cost�\�;l�9�+       ��K	�BU��A�*

logging/current_cost@H�;6�\+       ��K	�uU��A�*

logging/current_cost|1�;�x�g+       ��K	��U��A�*

logging/current_cost~�;F@�+       ��K	��U��A�*

logging/current_cost�	�;����+       ��K	qV��A�*

logging/current_cost���;�b7f+       ��K	_JV��A�*

logging/current_costn��;��U+       ��K	V��A�*

logging/current_cost���;ۑ�q+       ��K	��V��A�*

logging/current_costR��;��T�+       ��K	�V��A�*

logging/current_cost��;1��_+       ��K	�W��A�*

logging/current_costR��;�-�Q+       ��K	"HW��A�*

logging/current_cost���;��|+       ��K	�xW��A�*

logging/current_costt��;���N+       ��K	��W��A�*

logging/current_cost5t�;/���+       ��K	��W��A�*

logging/current_cost�c�;m/�B+       ��K	�X��A�*

logging/current_cost�Q�;�t`+       ��K	�;X��A�*

logging/current_cost�=�;���+       ��K	zkX��A�*

logging/current_costY%�;�G��+       ��K	��X��A�*

logging/current_cost��;�\��+       ��K	7�X��A�*

logging/current_costu �;9U4+       ��K	�Y��A�*

logging/current_cost��;��%@+       ��K	�;Y��A�*

logging/current_cost���;+�+       ��K	�nY��A�*

logging/current_cost���;�?H+       ��K	��Y��A�*

logging/current_costl��;Z7b�+       ��K	5�Y��A�*

logging/current_cost4��;'�+       ��K	�Z��A�*

logging/current_cost˩�;Ĺ�+       ��K	@FZ��A�*

logging/current_costܙ�;؆I�+       ��K	L{Z��A�*

logging/current_costĕ�;���+       ��K	G�Z��A�*

logging/current_cost��;Q.��+       ��K	t�Z��A�*

logging/current_cost`{�;���d+       ��K	[��A�*

logging/current_cost>v�;��he+       ��K	D[��A�*

logging/current_costij�;�ay[+       ��K	�x[��A�*

logging/current_cost'_�;�Z�y+       ��K	��[��A�*

logging/current_cost+Y�;�lH�+       ��K	��[��A�*

logging/current_cost�m�;�\$5+       ��K	�\��A�*

logging/current_cost�a�;7��+       ��K	�U\��A�*

logging/current_cost�E�;��n+       ��K	��\��A�*

logging/current_cost;A�;;ݥ+       ��K	A�\��A�*

logging/current_cost�?�;ߔ��+       ��K	��\��A�*

logging/current_cost�-�;s�JE+       ��K	�&]��A�*

logging/current_cost��;'3Q�+       ��K	�\]��A�*

logging/current_cost��;mIaP+       ��K	c�]��A�*

logging/current_cost�
�;�C++       ��K	��]��A�*

logging/current_cost���;3��+       ��K	�]��A�*

logging/current_cost��;����+       ��K	)^��A�*

logging/current_cost`��;�Ѐf+       ��K	p]^��A�*

logging/current_cost���;-��n+       ��K	ґ^��A�*

logging/current_cost ��;����+       ��K	�^��A�*

logging/current_cost+��;�9M�+       ��K	J�^��A�*

logging/current_costK��;s*Hl+       ��K	~*_��A�*

logging/current_costº�;C_�+       ��K	�\_��A�*

logging/current_cost˰�;\N�+       ��K	�_��A�*

logging/current_cost>��;��%�+       ��K	��_��A�*

logging/current_cost ��;P�:�+       ��K	\�_��A�*

logging/current_cost|��;�f��+       ��K	�,`��A�*

logging/current_cost���;���+       ��K	l^`��A�*

logging/current_cost���;� �+       ��K	�`��A�*

logging/current_cost܎�;�r2a+       ��K	��`��A�*

logging/current_cost ��;xX�u+       ��K	5�`��A�*

logging/current_cost�~�;8��+       ��K	�*a��A�*

logging/current_cost~�;�Ŏ+       ��K	�ua��A�*

logging/current_cost�s�;a�A+       ��K	e�a��A�*

logging/current_cost j�;��z+       ��K	{�a��A�*

logging/current_costm�; )]+       ��K	Rb��A�*

logging/current_cost�f�;�ե�+       ��K	�Pb��A�*

logging/current_costK\�;�eXz+       ��K	^�b��A�*

logging/current_cost�Z�;��S�+       ��K	H�b��A�	*

logging/current_cost�R�;��KF+       ��K	�c��A�	*

logging/current_cost�N�;�e+       ��K	Oc��A�	*

logging/current_cost5N�;�H4+       ��K	.�c��A�	*

logging/current_costlF�;���+       ��K	Ƿc��A�	*

logging/current_cost�@�;��c+       ��K	��c��A�	*

logging/current_cost�=�;��K+       ��K	�(d��A�	*

logging/current_cost�A�;�ۋ�+       ��K	ad��A�	*

logging/current_costU;�;�f�~+       ��K	��d��A�	*

logging/current_cost�2�;�rT+       ��K	[�d��A�	*

logging/current_cost�0�;�>[o+       ��K	��d��A�	*

logging/current_cost�-�;�[�+       ��K	�,e��A�	*

logging/current_cost�%�;$T!+       ��K	<ae��A�	*

logging/current_costG&�;��+       ��K	&�e��A�	*

logging/current_cost$�;.)��+       ��K	!�e��A�	*

logging/current_cost��;s%d+       ��K	1�e��A�	*

logging/current_costn�;R�<�+       ��K	-f��A�	*

logging/current_cost��;�l&+       ��K	�]f��A�	*

logging/current_cost@�;[+       ��K	��f��A�	*

logging/current_cost��;���+       ��K	��f��A�	*

logging/current_cost��;�+�+       ��K	��f��A�	*

logging/current_costN�;�Ƒ+       ��K	�5g��A�	*

logging/current_cost���;�<f+       ��K	�jg��A�	*

logging/current_cost<�;I�#�+       ��K	��g��A�	*

logging/current_cost9F�;� ��+       ��K	W�g��A�	*

logging/current_cost�/�;�u+       ��K	�h��A�
*

logging/current_cost��;) ��+       ��K	U8h��A�
*

logging/current_cost� �;Ʌ"�+       ��K	llh��A�
*

logging/current_costw)�;�?i+       ��K	�h��A�
*

logging/current_costi	�;}?+       ��K	��h��A�
*

logging/current_cost{	�;v�H+       ��K	�i��A�
*

logging/current_costn6�;5�<�+       ��K	<i��A�
*

logging/current_cost��;�J�j+       ��K	�pi��A�
*

logging/current_cost��;`T�+       ��K	��i��A�
*

logging/current_cost�;��0�+       ��K	"�i��A�
*

logging/current_cost��;F�!�+       ��K	�j��A�
*

logging/current_costt��;[�c�+       ��K	>j��A�
*

logging/current_cost7��;�k�q+       ��K	Roj��A�
*

logging/current_costL��;hnYA+       ��K	g�j��A�
*

logging/current_costҠ�;�u�+       ��K	�j��A�
*

logging/current_cost��;����+       ��K	�k��A�
*

logging/current_cost�c�;,�K+       ��K	7Pk��A�
*

logging/current_cost�A�;@�_+       ��K	ޅk��A�
*

logging/current_cost��;�Y��+       ��K	��k��A�
*

logging/current_cost�P�;�O�+       ��K	��k��A�
*

logging/current_cost<��;�;o�+       ��K	�l��A�
*

logging/current_cost[��;��<�+       ��K	(Rl��A�
*

logging/current_cost5�;b'#+       ��K	��l��A�
*

logging/current_costrJ�;�0�>+       ��K	��l��A�
*

logging/current_cost��;���+       ��K	��l��A�
*

logging/current_cost���;	�1#+       ��K	F!m��A�
*

logging/current_cost���;4'��+       ��K	�Rm��A�*

logging/current_cost��;øu�+       ��K	�m��A�*

logging/current_cost.��;��Vj+       ��K	�m��A�*

logging/current_costI��;0Q�F+       ��K	j�m��A�*

logging/current_cost�;�|��+       ��K	�n��A�*

logging/current_cost<�;�� �+       ��K	Dn��A�*

logging/current_cost ;�;bisO+       ��K	(un��A�*

logging/current_cost�D�;��3+       ��K	��n��A�*

logging/current_cost�+�;~2�+       ��K	��n��A�*

logging/current_costE�;�T��+       ��K	�o��A�*

logging/current_cost��;_�I�+       ��K	�Fo��A�*

logging/current_cost���;X:n�+       ��K	axo��A�*

logging/current_cost'��;s��+       ��K	E�o��A�*

logging/current_cost���;���+       ��K	Z�o��A�*

logging/current_cost�;B�{+       ��K	�p��A�*

logging/current_costU�;��+       ��K	�?p��A�*

logging/current_costN�;e�5+       ��K	�rp��A�*

logging/current_cost��;�FL+       ��K	$�p��A�*

logging/current_cost��;�u�@+       ��K	��p��A�*

logging/current_cost7�;�Z�+       ��K	�q��A�*

logging/current_cost �;v���+       ��K	M9q��A�*

logging/current_cost(�;)���+       ��K	�jq��A�*

logging/current_cost:�;����+       ��K	N�q��A�*

logging/current_cost�?�;����+       ��K		�q��A�*

logging/current_costU9�;�@��+       ��K	��q��A�*

logging/current_cost>2�;�
^|+       ��K	�1r��A�*

logging/current_cost""�;l�+       ��K	Bbr��A�*

logging/current_cost|�;�4i�+       ��K	��r��A�*

logging/current_cost�;� �+       ��K	Z�r��A�*

logging/current_cost���;_�M2+       ��K	��r��A�*

logging/current_cost���;K�֑+       ��K	X$s��A�*

logging/current_cost���;1<K+       ��K	�Xs��A�*

logging/current_cost���;�P�e+       ��K	 �s��A�*

logging/current_costb��;��oE+       ��K	��s��A�*

logging/current_cost���;�D�D+       ��K	\�s��A�*

logging/current_cost���;�|Y+       ��K	,,t��A�*

logging/current_costu��;���+       ��K	m_t��A�*

logging/current_cost�'�;�i��+       ��K	��t��A�*

logging/current_cost��;����+       ��K	U�t��A�*

logging/current_cost���;�k��+       ��K	�t��A�*

logging/current_cost���;�38+       ��K	t&u��A�*

logging/current_cost���;�-�+       ��K	h\u��A�*

logging/current_cost`��;,���+       ��K	��u��A�*

logging/current_cost|��;�u�+       ��K	l�u��A�*

logging/current_costK��;�z�+       ��K	6�u��A�*

logging/current_cost���;�8ז+       ��K	�&v��A�*

logging/current_cost���;;Q�+       ��K	�Xv��A�*

logging/current_cost+��;����+       ��K	�v��A�*

logging/current_costD}�;�No�+       ��K	w�v��A�*

logging/current_cost,t�;�j�>+       ��K	��v��A�*

logging/current_costyz�;}��/+       ��K	�/w��A�*

logging/current_cost��;e�L�+       ��K	~dw��A�*

logging/current_cost�t�;����+       ��K	�w��A�*

logging/current_cost�T�;���Y+       ��K	B�w��A�*

logging/current_cost":�;�x�
+       ��K	��w��A�*

logging/current_cost:�;�
8{+       ��K	[(x��A�*

logging/current_costD9�;D���+       ��K	3Xx��A�*

logging/current_cost�(�;;na}+       ��K	�x��A�*

logging/current_cost4_�;�*k+       ��K	��x��A�*

logging/current_cost�>�;x���+       ��K	��x��A�*

logging/current_cost�H�;5�! +       ��K	�y��A�*

logging/current_cost3�;��?t+       ��K	><y��A�*

logging/current_cost�(�;�C�+       ��K	Fiy��A�*

logging/current_cost��;Y/tH+       ��K	��y��A�*

logging/current_costg&�;e�l+       ��K	��y��A�*

logging/current_costg+�;J	��+       ��K	W�y��A�*

logging/current_cost.C�;f�`"+       ��K	(#z��A�*

logging/current_costPE�;�+       ��K	�Pz��A�*

logging/current_costRD�;�*�x+       ��K	�}z��A�*

logging/current_cost�>�;�[.X+       ��K	
�z��A�*

logging/current_cost�.�;�*B+       ��K	��z��A�*

logging/current_cost.*�;�f)�+       ��K	�{��A�*

logging/current_cost� �;_���+       ��K	I8{��A�*

logging/current_cost��;�r�+       ��K	�d{��A�*

logging/current_cost��;z��h+       ��K	d�{��A�*

logging/current_cost��;+lM�+       ��K	%|��A�*

logging/current_cost��;r��@+       ��K	�=|��A�*

logging/current_cost��;a�+       ��K	I�|��A�*

logging/current_cost�;�=��+       ��K	��|��A�*

logging/current_cost��;�<�+       ��K	c�|��A�*

logging/current_cost��;60+       ��K	�#}��A�*

logging/current_cost\�;o���+       ��K	JY}��A�*

logging/current_cost.�;7,�+       ��K	F�}��A�*

logging/current_costp
�;����+       ��K	��}��A�*

logging/current_cost��;��P�+       ��K	��}��A�*

logging/current_cost��;fK
+       ��K	7"~��A�*

logging/current_cost��;;؏+       ��K	eY~��A�*

logging/current_cost~�;ޕo�+       ��K	�~��A�*

logging/current_cost���;\9+       ��K	��~��A�*

logging/current_costK��;��^w+       ��K	�~��A�*

logging/current_cost���;�_+�+       ��K	���A�*

logging/current_cost;��;��+       ��K	�C��A�*

logging/current_cost���;B<�+       ��K	�q��A�*

logging/current_costr��;��,+       ��K	l���A�*

logging/current_costr��;�q�+       ��K	����A�*

logging/current_cost���;��,�+       ��K	����A�*

logging/current_cost���;ﴛ�+       ��K	�?���A�*

logging/current_costL��;MKv"+       ��K	�|���A�*

logging/current_cost���;db��+       ��K	o����A�*

logging/current_costI��;��A+       ��K	w���A�*

logging/current_cost��;Kb�+       ��K	]C���A�*

logging/current_cost���;^��+       ��K	�{���A�*

logging/current_cost���;��8+       ��K	סּ��A�*

logging/current_costy��;�-��+       ��K	)⁓�A�*

logging/current_cost���;?���+       ��K	W���A�*

logging/current_costk��;�)�+       ��K	_���A�*

logging/current_cost@��;�q:�+       ��K	�����A�*

logging/current_cost���;��`+       ��K	{˂��A�*

logging/current_costп�;y�u+       ��K	 ���A�*

logging/current_cost���;�A�+       ��K	�?���A�*

logging/current_cost���;z�`d+       ��K	z���A�*

logging/current_cost���;5�1`+       ��K	䲃��A�*

logging/current_cost���;�0��+       ��K	�僓�A�*

logging/current_cost���;ǜ~+       ��K	���A�*

logging/current_cost��;��v�+       ��K	Q���A�*

logging/current_cost@��;����+       ��K	ȃ���A�*

logging/current_cost'��;�ҳz+       ��K	�����A�*

logging/current_cost2��;�>+       ��K	�焓�A�*

logging/current_cost˴�;a�+       ��K	^����A�*

logging/current_cost��;�1eT+       ��K	�����A�*

logging/current_cost��;��+       ��K	��A�*

logging/current_cost��;�=�+       ��K	�,���A�*

logging/current_costҴ�;l�1+       ��K	�`���A�*

logging/current_cost���;�-w�+       ��K	����A�*

logging/current_cost���;Zj�9+       ��K	�͆��A�*

logging/current_cost���;�\B�+       ��K	���A�*

logging/current_cost���;L �+       ��K	]8���A�*

logging/current_cost4��;�䴿+       ��K	5e���A�*

logging/current_cost.��;���+       ��K	M����A�*

logging/current_cost��;:��+       ��K	Ͽ���A�*

logging/current_costĲ�;�Q��+       ��K	����A�*

logging/current_cost\��;��r+       ��K	����A�*

logging/current_costk��;�m��+       ��K	FL���A�*

logging/current_costi��;Z�p�+       ��K	cy���A�*

logging/current_cost���;۫}�+       ��K	⫈��A�*

logging/current_costR��;�=��+       ��K	jވ��A�*

logging/current_cost���;�}>�+       ��K	����A�*

logging/current_cost5��;C6�O+       ��K	`:���A�*

logging/current_costB��;B+       ��K	�h���A�*

logging/current_costצ�;��
�+       ��K	�����A�*

logging/current_cost���;��q+       ��K	�ŉ��A�*

logging/current_costǤ�;N�\+       ��K	���A�*

logging/current_cost��;ߡ��+       ��K	�#���A�*

logging/current_cost���;���O+       ��K	�P���A�*

logging/current_cost$��;ZF��+       ��K	;~���A�*

logging/current_cost���;��+       ��K	쫊��A�*

logging/current_cost$��;F�G�+       ��K	Xڊ��A�*

logging/current_costհ�;y��+       ��K	�	���A�*

logging/current_cost���;�Kh�+       ��K	�7���A�*

logging/current_cost7��;2d+       ��K	f���A�*

logging/current_cost���;���+       ��K	9����A�*

logging/current_cost���;�u�s+       ��K	zċ��A�*

logging/current_cost���;�+j�+       ��K	S��A�*

logging/current_costף�;�s�+       ��K	�$���A�*

logging/current_cost���;��+       ��K	�Z���A�*

logging/current_cost��;�-�"+       ��K	8����A�*

logging/current_cost���;u�=�+       ��K	����A�*

logging/current_cost���;����+       ��K	[猓�A�*

logging/current_cost+��;\���+       ��K	M!���A�*

logging/current_costҘ�;0a�R+       ��K	�T���A�*

logging/current_cost���;�:�+       ��K	S����A�*

logging/current_cost7��;��W�+       ��K	S����A�*

logging/current_cost���;L�-7+       ��K	捓�A�*

logging/current_cost٢�;���+       ��K	����A�*

logging/current_cost5��;�C++       ��K	G���A�*

logging/current_costD��;/<�`+       ��K	�x���A�*

logging/current_cost���;r�A�+       ��K	많��A�*

logging/current_cost��;jum>+       ��K	�Ҏ��A�*

logging/current_cost ��;�
��+       ��K	����A�*

logging/current_cost��;�Q+       ��K	E2���A�*

logging/current_cost���;[q�+       ��K	(c���A�*

logging/current_cost���;oQ��+       ��K	协��A�*

logging/current_cost���;�4�+       ��K	̺���A�*

logging/current_cost|��;k���+       ��K	Cꏓ�A�*

logging/current_cost>��;Q��+       ��K	U���A�*

logging/current_cost���;�τ+       ��K	�H���A�*

logging/current_cost���;��^�+       ��K	�x���A�*

logging/current_costg��;���B+       ��K	�����A�*

logging/current_cost4��;b+�)+       ��K	�א��A�*

logging/current_cost@��;W~Q*+       ��K	����A�*

logging/current_cost��;�'��+       ��K	^5���A�*

logging/current_cost���;B�y+       ��K	�e���A�*

logging/current_cost� �;L�Z+       ��K	U����A�*

logging/current_cost�!�;(��O+       ��K	*Ñ��A�*

logging/current_cost��;�]#+       ��K	��A�*

logging/current_cost��;2�Y�+       ��K	� ���A�*

logging/current_cost���;�Z�<+       ��K	�O���A�*

logging/current_coste�;���+       ��K	1}���A�*

logging/current_cost��;�-�+       ��K	����A�*

logging/current_costg�;��R�+       ��K	�ݒ��A�*

logging/current_cost$�;r�	�+       ��K	o���A�*

logging/current_costR�;���+       ��K	�:���A�*

logging/current_cost,�;�8�U+       ��K	�h���A�*

logging/current_cost�9�;|�W+       ��K	�����A�*

logging/current_cost�N�;""�+       ��K	!͓��A�*

logging/current_cost\�;��L+       ��K	�����A�*

logging/current_cost�n�;W�ʡ+       ��K	�,���A�*

logging/current_cost y�;�S��+       ��K	�Y���A�*

logging/current_cost��;�=�+       ��K	�����A�*

logging/current_costT��;��t�+       ��K	!����A�*

logging/current_cost���;ϸ�+       ��K	|唓�A�*

logging/current_cost���;,oGQ+       ��K	,���A�*

logging/current_cost���;��q+       ��K	;>���A�*

logging/current_cost���;0�h+       ��K	�m���A�*

logging/current_costd��;{��+       ��K	K����A�*

logging/current_cost���;�\��+       ��K	%˕��A�*

logging/current_cost ��;
�+       ��K	�����A�*

logging/current_cost7��;-�*�+       ��K	,$���A�*

logging/current_cost@��;!c�+       ��K	�N���A�*

logging/current_cost��;��{A+       ��K	A}���A�*

logging/current_costB��; �+       ��K	�����A�*

logging/current_costے�;P�o�+       ��K	2ؖ��A�*

logging/current_costD��;*$�H+       ��K	J���A�*

logging/current_cost5��;i�M+       ��K	�/���A�*

logging/current_costg��;��\�+       ��K	�Z���A�*

logging/current_costً�;g���+       ��K	`����A�*

logging/current_costɉ�;�j�+       ��K	~����A�*

logging/current_cost\��;	�Р+       ��K	�藓�A�*

logging/current_cost���;`���+       ��K	���A�*

logging/current_costk��;�<&p+       ��K	u?���A�*

logging/current_cost���;�&Ы+       ��K	m���A�*

logging/current_cost�~�;[���+       ��K	^����A�*

logging/current_cost΀�;(�h+       ��K	KƘ��A�*

logging/current_costy�;��*n+       ��K	���A�*

logging/current_cost|�;����+       ��K	����A�*

logging/current_cost�}�;��c�+       ��K	P���A�*

logging/current_cost||�;<�b�+       ��K	U����A�*

logging/current_cost |�;���+       ��K	B����A�*

logging/current_cost'{�;�b�+       ��K	xܙ��A�*

logging/current_costRx�;��ӳ+       ��K	[	���A�*

logging/current_cost�x�;_宭+       ��K	z7���A�*

logging/current_cost�v�;K�k+       ��K	�d���A�*

logging/current_cost.u�;��^+       ��K	5����A�*

logging/current_cost�o�;w�Q�+       ��K	�Ś��A�*

logging/current_cost5a�;p��d+       ��K	�����A�*

logging/current_costRL�;���+       ��K	!#���A�*

logging/current_cost>>�;z!�+       ��K	�O���A�*

logging/current_cost�8�;��u�+       ��K	�|���A�*

logging/current_costP5�;����+       ��K	�����A�*

logging/current_cost�,�;���+       ��K	�����A�*

logging/current_cost�+�;�ak�+       ��K	����A�*

logging/current_cost�-�;-��B+       ��K	�A���A�*

logging/current_costg5�;�#�j+       ��K	u���A�*

logging/current_cost�H�;�f[+       ��K	Ϥ���A�*

logging/current_cost�\�;[��+       ��K	�Ҝ��A�*

logging/current_cost�h�;\hUy+       ��K	c���A�*

logging/current_cost�d�;%�7+       ��K	�.���A�*

logging/current_cost�I�;���v+       ��K	�\���A�*

logging/current_cost�.�;��G)+       ��K	�����A�*

logging/current_cost�"�;G��?+       ��K	����A�*

logging/current_cost��;"lQ+       ��K	P흓�A�*

logging/current_cost)�;�em+       ��K	����A�*

logging/current_cost���;�am+       ��K	WG���A�*

logging/current_cost���;GoD+       ��K	�}���A�*

logging/current_cost���;���+       ��K	Ƭ���A�*

logging/current_cost@��;��I+       ��K	�ٞ��A�*

logging/current_cost5��;V+��+       ��K	����A�*

logging/current_cost���;a��{+       ��K	�7���A�*

logging/current_cost���;I�y+       ��K	pe���A�*

logging/current_cost���;�<�+       ��K	}����A�*

logging/current_cost���;	���+       ��K	�ҟ��A�*

logging/current_cost��;J��+       ��K	����A�*

logging/current_costY��;���+       ��K	^/���A�*

logging/current_cost¿�;�NH+       ��K	?[���A�*

logging/current_cost���;P�W+       ��K	ٌ���A�*

logging/current_cost���;��s+       ��K	Z����A�*

logging/current_cost���;sD�+       ��K	렓�A�*

logging/current_cost|��;�0,+       ��K	����A�*

logging/current_costY��;qL��+       ��K	PF���A�*

logging/current_cost��;R���+       ��K	s���A�*

logging/current_cost���;%�+       ��K	ؤ���A�*

logging/current_cost���;	 �?+       ��K	ѡ��A�*

logging/current_costu��;����+       ��K	j����A�*

logging/current_cost0��;_d+       ��K	�*���A�*

logging/current_cost`��;�!��+       ��K	�W���A�*

logging/current_costΫ�;�;+       ��K	�����A�*

logging/current_cost���;��1+       ��K	,����A�*

logging/current_costէ�;�&*�+       ��K	|ޢ��A�*

logging/current_cost`��;��z�+       ��K	y
���A�*

logging/current_costk��;�t��+       ��K	�9���A�*

logging/current_cost²�;���+       ��K	'h���A�*

logging/current_cost���;#4�H+       ��K	Y����A�*

logging/current_cost���;�A�T+       ��K	ã��A�*

logging/current_costn��;CU�+       ��K	(��A�*

logging/current_cost���;�p�+       ��K	���A�*

logging/current_costW��;���+       ��K	�N���A�*

logging/current_cost��;�'x+       ��K	d}���A�*

logging/current_cost��;�ֵ+       ��K	�����A�*

logging/current_cost���; �U�+       ��K	פ��A�*

logging/current_costě�;!g�+       ��K	B���A�*

logging/current_cost��;���m+       ��K	�0���A�*

logging/current_cost9��;Td<+       ��K	�^���A�*

logging/current_cost���;�Hy�+       ��K	�����A�*

logging/current_cost���;��+       ��K	񹥓�A�*

logging/current_cost���;���E+       ��K	楓�A�*

logging/current_cost2�;�In�+       ��K	~���A�*

logging/current_cost L�;��ç+       ��K	zC���A�*

logging/current_costNl�;jxL+       ��K	�p���A�*

logging/current_costj�;0�w�+       ��K	h����A�*

logging/current_cost�a�;�?�+       ��K	�ͦ��A�*

logging/current_cost�b�;/�+       ��K	�����A�*

logging/current_cost�I�;�H�O+       ��K	u+���A�*

logging/current_cost�)�;��c�+       ��K	[\���A�*

logging/current_cost�)�;k8�3+       ��K	�����A�*

logging/current_cost�.�;�hp+       ��K	�����A�*

logging/current_costw3�;�E+       ��K	X�A�*

logging/current_cost�1�;+P��+       ��K	����A�*

logging/current_costB-�;�2�+       ��K	�K���A�*

logging/current_cost@$�;��%+       ��K	�x���A�*

logging/current_cost�!�;�'�+       ��K	�����A�*

logging/current_cost"#�;t�+       ��K	>ר��A�*

logging/current_cost�!�;l4��+       ��K	D���A�*

logging/current_cost�#�;�v�+       ��K	�2���A�*

logging/current_cost�;�eV�+       ��K	d���A�*

logging/current_costu �;K>1*+       ��K	ђ���A�*

logging/current_cost9�;�}�W+       ��K	M����A�*

logging/current_cost��;W�6+       ��K	�禎�A�*

logging/current_cost��;[��Z+       ��K	� ���A�*

logging/current_cost��;=��+       ��K	N���A�*

logging/current_cost��;7���+       ��K	�z���A�*

logging/current_cost� �;O�#v+       ��K	�����A�*

logging/current_cost��;Zp�_+       ��K	]ܪ��A�*

logging/current_cost��;XV��+       ��K	s
���A�*

logging/current_costd�;��O+       ��K	.9���A�*

logging/current_cost��;_X+       ��K	$e���A�*

logging/current_cost��;@�;2+       ��K	����A�*

logging/current_cost��;5��0+       ��K	\ƫ��A�*

logging/current_costd�;��U�+       ��K	5����A�*

logging/current_cost�;;h�P+       ��K	W#���A�*

logging/current_cost��;�ވ^+       ��K	9W���A�*

logging/current_cost��;50l+       ��K	�����A�*

logging/current_cost�; Hh}+       ��K	)����A�*

logging/current_cost2 �;���+       ��K	ᬓ�A�*

logging/current_cost+�;
N+       ��K	����A�*

logging/current_costY�;]pO+       ��K	�<���A�*

logging/current_cost2(�;�N=�+       ��K	5k���A�*

logging/current_cost��;�;�+       ��K	[����A�*

logging/current_cost�;og�+       ��K	˭��A�*

logging/current_cost��;��"+       ��K	�����A�*

logging/current_cost��;dj��+       ��K	�'���A�*

logging/current_cost��;?T��+       ��K	W���A�*

logging/current_cost���;Fܽ�+       ��K	m����A�*

logging/current_cost���;�n��+       ��K	����A�*

logging/current_cost��;����+       ��K	?߮��A�*

logging/current_cost�
�;]�2p+       ��K	���A�*

logging/current_cost@�;�K�+       ��K	t=���A�*

logging/current_cost���;H��E+       ��K	cl���A�*

logging/current_cost��;,Jj+       ��K	�����A�*

logging/current_cost �;���+       ��K	�ʯ��A�*

logging/current_cost4��;�,K&+       ��K	|����A�*

logging/current_costB�;u�+       ��K	f&���A�*

logging/current_cost��;��K�+       ��K	�V���A�*

logging/current_cost��;�@X+       ��K	k����A�*

logging/current_cost���;TЃ�+       ��K	#����A�*

logging/current_cost'��;Q��+       ��K	Cް��A�*

logging/current_cost���;$�h+       ��K	����A�*

logging/current_cost� �;�L+       ��K	�9���A�*

logging/current_cost��;�ٟ�+       ��K	^g���A�*

logging/current_costl=�;��r�+       ��K	B����A�*

logging/current_costYB�;�+�k+       ��K	2����A�*

logging/current_cost	*�;��O�+       ��K	�����A�*

logging/current_cost�*�;[��+       ��K	����A�*

logging/current_cost8�;>_��+       ��K	�H���A�*

logging/current_cost 7�;��+       ��K	hv���A�*

logging/current_cost^O�;���+       ��K	�����A�*

logging/current_cost�H�;��+       ��K	�Ҳ��A�*

logging/current_cost�]�;yb��+       ��K	Y����A�*

logging/current_cost|o�;��J+       ��K	�+���A�*

logging/current_cost���;�(d+       ��K	�[���A�*

logging/current_costg��;�0K�+       ��K	关��A�*

logging/current_cost�m�;�5�%+       ��K	�����A�*

logging/current_cost+w�;�B0+       ��K	�೓�A�*

logging/current_cost�m�;KBX@+       ��K	Y���A�*

logging/current_cost ��;Ld�i+       ��K	>���A�*

logging/current_cost���;�Հ+       ��K	Cl���A�*

logging/current_cost���;o��+       ��K	�����A�*

logging/current_costē�;��4�+       ��K	�Ŵ��A�*

logging/current_cost�;���0+       ��K	�����A�*

logging/current_cost���;���E+       ��K	�*���A�*

logging/current_costU��;6�1+       ��K	[���A�*

logging/current_costG��;UN%;+       ��K	懵��A�*

logging/current_costܑ�;�<!+       ��K	ܵ���A�*

logging/current_cost���;�fI+       ��K	�ⵓ�A�*

logging/current_costܔ�;�@�+       ��K	���A�*

logging/current_cost��;����+       ��K	z?���A�*

logging/current_costy��;���%+       ��K	:r���A�*

logging/current_cost��;��(+       ��K	����A�*

logging/current_cost�|�;fUW�+       ��K	Ѷ��A�*

logging/current_costw��;{B�b+       ��K	�����A�*

logging/current_cost ��;��Ph+       ��K	.,���A�*

logging/current_cost׋�;�(�G+       ��K	�Z���A�*

logging/current_costy��;�Q(+       ��K	�����A�*

logging/current_cost ��;��
+       ��K	׺���A�*

logging/current_cost���;��+       ��K	췓�A�*

logging/current_costw��;i�1:+       ��K	K���A�*

logging/current_cost���;�>�+       ��K	gI���A�*

logging/current_cost���;~.��+       ��K	�v���A�*

logging/current_cost���;�<||+       ��K	M����A�*

logging/current_cost`��;g�4+       ��K	�ո��A�*

logging/current_cost��;��=+       ��K	���A�*

logging/current_cost���;)��3+       ��K	C1���A�*

logging/current_costd��;8v�+       ��K	�_���A�*

logging/current_costk��;�׬1+       ��K	�����A�*

logging/current_cost��;�U��+       ��K	����A�*

logging/current_costY��;��;+       ��K	E繓�A�*

logging/current_costΡ�;�C�
+       ��K	����A�*

logging/current_cost+��;�C��+       ��K	�C���A�*

logging/current_cost ��;Z�`�+       ��K	�p���A�*

logging/current_cost ��;[�i+       ��K	�����A�*

logging/current_cost���;Sٰ-+       ��K	"Ѻ��A�*

logging/current_cost ��;�w��+       ��K	� ���A�*

logging/current_cost��;���+       ��K	�.���A�*

logging/current_cost���;y~�j+       ��K	�[���A�*

logging/current_costģ�;w<�+       ��K	C����A�*

logging/current_cost@��;�9�+       ��K	)��A�*

logging/current_cost��;^74W+       ��K	�*���A�*

logging/current_cost���;waU�+       ��K	Wg���A�*

logging/current_cost,��;�Q(�+       ��K	7����A�*

logging/current_cost��;:��+       ��K	�ݼ��A�*

logging/current_cost��;��+       ��K	����A�*

logging/current_costN��;�"`�+       ��K	Z���A�*

logging/current_costէ�;GT+       ��K	𕽓�A�*

logging/current_cost���;n5�+       ��K	�ҽ��A�*

logging/current_cost���;��D+       ��K	���A�*

logging/current_cost|��;? �+       ��K	�G���A�*

logging/current_cost���;���C+       ��K	�����A�*

logging/current_cost��;��%�+       ��K	�����A�*

logging/current_cost���;�0�+       ��K	��A�*

logging/current_cost	��;R�~b+       ��K	*-���A�*

logging/current_cost���;��O@+       ��K	�b���A�*

logging/current_cost��;I���+       ��K	�����A�*

logging/current_costҤ�;���C+       ��K	�̿��A�*

logging/current_cost���;���+       ��K	r���A�*

logging/current_costl��;�ۘ+       ��K	T3���A�*

logging/current_cost���;`g"+       ��K	�c���A�*

logging/current_cost��;��+       ��K	y����A�*

logging/current_cost��;��}+       ��K	�����A�*

logging/current_costt��;�s�+       ��K	#����A�*

logging/current_cost���;�h�+       ��K	�)���A�*

logging/current_cost��;>>+       ��K	�W���A�*

logging/current_costN��;�2+       ��K	!����A�*

logging/current_cost)��;���+       ��K	^����A�*

logging/current_costԝ�;�6��+       ��K	+����A�*

logging/current_cost���;�}��+       ��K	�A�*

logging/current_cost|��;X��+       ��K	�Q�A�*

logging/current_cost<��;��+       ��K	>��A�*

logging/current_costp��;�с�+       ��K	|��A�*

logging/current_cost	��;��G+       ��K	���A�*

logging/current_cost+��;� ��+       ��K	Ó�A�*

logging/current_costն�;E�:+       ��K	�LÓ�A�*

logging/current_cost���;k���+       ��K	�zÓ�A�*

logging/current_cost��;����+       ��K	��Ó�A�*

logging/current_costL��;��o�+       ��K	��Ó�A�*

logging/current_cost;��;��-+       ��K	ē�A�*

logging/current_cost|��;���/+       ��K	<=ē�A�*

logging/current_cost���;��I+       ��K	bnē�A�*

logging/current_cost��;r���+       ��K	�ē�A�*

logging/current_costU��;���+       ��K	q�ē�A�*

logging/current_cost���;LO��+       ��K	��ē�A�*

logging/current_cost���;� ��+       ��K	�)œ�A�*

logging/current_cost���;���+       ��K	�Wœ�A�*

logging/current_cost���;�d=+       ��K	��œ�A�*

logging/current_cost^��;���+       ��K	X�œ�A�*

logging/current_cost���;r�9f+       ��K	�œ�A�*

logging/current_costi��;��ۋ+       ��K	�"Ɠ�A�*

logging/current_cost4��;*�oH+       ��K	�RƓ�A�*

logging/current_cost��;����+       ��K	d�Ɠ�A�*

logging/current_cost���;�d֩+       ��K	ְƓ�A�*

logging/current_cost���;��~+       ��K	U�Ɠ�A�*

logging/current_cost'��;�h}�+       ��K	+Ǔ�A�*

logging/current_cost���;��"+       ��K	�>Ǔ�A�*

logging/current_cost���;���+       ��K	�kǓ�A�*

logging/current_cost;��;��I�+       ��K	כǓ�A�*

logging/current_cost���;���"+       ��K	��Ǔ�A�*

logging/current_costU��;����+       ��K	9�Ǔ�A�*

logging/current_cost ��;#�b�+       ��K	�&ȓ�A�*

logging/current_cost,��;�3�+       ��K	�Sȓ�A�*

logging/current_cost���;��+       ��K	�ȓ�A�*

logging/current_costY��;��Zu+       ��K	7�ȓ�A�*

logging/current_cost.��;�Il�+       ��K	��ȓ�A�*

logging/current_cost���;Mt�++       ��K	ɓ�A�*

logging/current_cost���;t�+       ��K	CCɓ�A�*

logging/current_costI��;a��+       ��K	irɓ�A�*

logging/current_cost9��;&4'�+       ��K	r�ɓ�A�*

logging/current_cost���;��
_+       ��K		�ɓ�A�*

logging/current_costg��;��0+       ��K	��ɓ�A�*

logging/current_cost ��;4��,+       ��K	X,ʓ�A�*

logging/current_cost���;�{@�+       ��K	!Xʓ�A�*

logging/current_cost���;��#d+       ��K	@�ʓ�A�*

logging/current_cost���;3�L+       ��K	*�ʓ�A�*

logging/current_cost��;Ǥ"�+       ��K	��ʓ�A�*

logging/current_cost��;�+       ��K	�˓�A�*

logging/current_cost���;�)i+       ��K	�=˓�A�*

logging/current_cost;��;�dL�+       ��K	�i˓�A�*

logging/current_cost���;��`+       ��K	5�˓�A�*

logging/current_cost���;��T�+       ��K	'�˓�A�*

logging/current_cost���;^�K+       ��K	�˓�A�*

logging/current_cost��;�f�+       ��K	�%̓�A�*

logging/current_cost��;����+       ��K	V̓�A�*

logging/current_cost`��;W*�+       ��K	��̓�A�*

logging/current_cost���;���l+       ��K	k�̓�A�*

logging/current_cost ��;���+       ��K	D�̓�A�*

logging/current_costn��;#��+       ��K	�#͓�A�*

logging/current_cost���;[�^�+       ��K	�^͓�A�*

logging/current_cost5��;�f��+       ��K	��͓�A�*

logging/current_cost���;���&+       ��K	��͓�A�*

logging/current_cost���;�P+       ��K	%Γ�A�*

logging/current_cost���;ݣ�+       ��K	YΓ�A�*

logging/current_cost���;�7�w+       ��K	��Γ�A�*

logging/current_costB��;�C�+       ��K	��Γ�A�*

logging/current_cost���;�G�+       ��K	Wϓ�A�*

logging/current_cost���;��`�+       ��K	TAϓ�A�*

logging/current_costl��;�gM�+       ��K	�sϓ�A�*

logging/current_cost��;mDe�+       ��K	��ϓ�A�*

logging/current_cost���;M!��+       ��K	!�ϓ�A�*

logging/current_cost���;=hu�+       ��K	PГ�A�*

logging/current_cost+��;$=��+       ��K	�NГ�A�*

logging/current_cost���;��e�+       ��K	��Г�A�*

logging/current_cost�;�V`+       ��K	z�Г�A�*

logging/current_cost� �;��P+       ��K	��Г�A�*

logging/current_costt�;�5�>+       ��K	8!ѓ�A�*

logging/current_costG��;;4�)+       ��K	�fѓ�A�*

logging/current_cost���;�.��+       ��K	ߘѓ�A�*

logging/current_cost��;�Sj{+       ��K	�ѓ�A�*

logging/current_cost<�;6է+       ��K	+�ѓ�A�*

logging/current_cost.�;�+A}+       ��K	�1ғ�A�*

logging/current_cost5
�;���+       ��K	rdғ�A�*

logging/current_costD�;�Dt+       ��K	�ғ�A�*

logging/current_costy�;If�+       ��K	k�ғ�A�*

logging/current_costI �;ב�+       ��K	ӓ�A�*

logging/current_cost��;b�Qb+       ��K	�6ӓ�A�*

logging/current_costD�;��U+       ��K	�mӓ�A�*

logging/current_cost��;�e�+       ��K	��ӓ�A�*

logging/current_cost���;ho-�+       ��K	�ԓ�A�*

logging/current_cost`�;ށ�+       ��K	�5ԓ�A�*

logging/current_cost�
�;�+�*+       ��K	4cԓ�A�*

logging/current_cost��;֍�D+       ��K	=�ԓ�A�*

logging/current_cost'�;L?�+       ��K	��ԓ�A�*

logging/current_cost �;h3#�+       ��K	u�ԓ�A� *

logging/current_cost�;WlG'+       ��K	�Փ�A� *

logging/current_cost�;0 U+       ��K	�KՓ�A� *

logging/current_cost^�;>���+       ��K	�Փ�A� *

logging/current_cost;�;)���+       ��K	�Փ�A� *

logging/current_cost�;/�ҳ+       ��K	��Փ�A� *

logging/current_cost �;-�L+       ��K	�'֓�A� *

logging/current_cost��;�:��+       ��K	�c֓�A� *

logging/current_costI!�;�[��+       ��K	��֓�A� *

logging/current_costU�;͖]�+       ��K	}�֓�A� *

logging/current_cost��;�|+       ��K	� ד�A� *

logging/current_costP!�;iJ�+       ��K	�1ד�A� *

logging/current_costD�;)4~�+       ��K	�jד�A� *

logging/current_costY"�;GB6+       ��K	��ד�A� *

logging/current_costK�;���+       ��K	u�ד�A� *

logging/current_cost�;�w
+       ��K	�ד�A� *

logging/current_costY�;E�+       ��K	�0ؓ�A� *

logging/current_cost.�;�l�+       ��K	�^ؓ�A� *

logging/current_costk�;W�+       ��K	�ؓ�A� *

logging/current_cost��;L��/+       ��K	߻ؓ�A� *

logging/current_cost��;]յ�+       ��K	�ؓ�A� *

logging/current_cost��;�L]�+       ��K	�ٓ�A� *

logging/current_cost��;]f�+       ��K	�Nٓ�A� *

logging/current_cost$�;Ɣe�+       ��K	B�ٓ�A� *

logging/current_costn�;xN�8+       ��K	��ٓ�A� *

logging/current_cost�%�;-�;+       ��K	��ٓ�A�!*

logging/current_cost�"�;Mڏ+       ��K	}ړ�A�!*

logging/current_cost��;;l��+       ��K	�Iړ�A�!*

logging/current_cost~�;�(�.+       ��K	�zړ�A�!*

logging/current_cost|�;�]��+       ��K	�ړ�A�!*

logging/current_cost��;憁�+       ��K	U�ړ�A�!*

logging/current_costg�;�T+       ��K	(ۓ�A�!*

logging/current_cost��;Ǿ��+       ��K	�4ۓ�A�!*

logging/current_cost,�;��{+       ��K	Ubۓ�A�!*

logging/current_costy!�;ζ�"+       ��K	u�ۓ�A�!*

logging/current_costP�;j�V�+       ��K	u�ۓ�A�!*

logging/current_costu �;���+       ��K	��ۓ�A�!*

logging/current_cost�;�?y�+       ��K	�*ܓ�A�!*

logging/current_cost� �;Z��+       ��K	�[ܓ�A�!*

logging/current_costY#�;��N\+       ��K	3�ܓ�A�!*

logging/current_cost$�;���+       ��K	��ܓ�A�!*

logging/current_cost�#�;Wt�+       ��K	��ܓ�A�!*

logging/current_costU�;i�|Q+       ��K	ݓ�A�!*

logging/current_cost��;���a+       ��K	�Hݓ�A�!*

logging/current_cost�;S��l+       ��K	#xݓ�A�!*

logging/current_cost��;���+       ��K	�ݓ�A�!*

logging/current_cost��;�U�?+       ��K	X�ݓ�A�!*

logging/current_cost$�;֗8-+       ��K	Dޓ�A�!*

logging/current_cost��;^nA&+       ��K	�3ޓ�A�!*

logging/current_cost� �;ۡe+       ��K	�cޓ�A�!*

logging/current_cost@�;{C��+       ��K	��ޓ�A�!*

logging/current_cost,�;l��+       ��K	)�ޓ�A�"*

logging/current_cost��;':�+       ��K	��ޓ�A�"*

logging/current_cost��;���I+       ��K	r ߓ�A�"*

logging/current_cost2�;G^Q�+       ��K	�Lߓ�A�"*

logging/current_cost�%�;��)U+       ��K	�ߓ�A�"*

logging/current_cost��;:h�+       ��K	�ߓ�A�"*

logging/current_cost�&�;��*+       ��K	��ߓ�A�"*

logging/current_cost��;1I+       ��K	����A�"*

logging/current_costD�;QgM�+       ��K	=���A�"*

logging/current_cost��;��5�+       ��K	k���A�"*

logging/current_cost+�;8��4+       ��K	u����A�"*

logging/current_cost^,�;��+       ��K	�����A�"*

logging/current_cost��;�V/+       ��K	l����A�"*

logging/current_costd+�;��.�+       ��K	k+��A�"*

logging/current_costY�;����+       ��K	�X��A�"*

logging/current_coste�;ݐ��+       ��K	���A�"*

logging/current_cost'�;�uy�+       ��K	+���A�"*

logging/current_cost��;%��+       ��K	����A�"*

logging/current_cost��;����+       ��K	]��A�"*

logging/current_costY�;��+       ��K	�A��A�"*

logging/current_cost�
�;0�+#+       ��K	in��A�"*

logging/current_cost��;3�D+       ��K	=���A�"*

logging/current_cost��;��+       ��K	����A�"*

logging/current_costN	�;�e~�+       ��K	S���A�"*

logging/current_cost��;N(��+       ��K	&��A�"*

logging/current_cost
�;($��+       ��K	�R��A�#*

logging/current_cost0�;SĽ�+       ��K	���A�#*

logging/current_cost�;�Rz,+       ��K	Ю��A�#*

logging/current_cost%��;gB4�+       ��K	"���A�#*

logging/current_cost>�;����+       ��K	k	��A�#*

logging/current_cost��;�F+       ��K	�6��A�#*

logging/current_cost0�;gc�+       ��K	�e��A�#*

logging/current_cost��;ո��+       ��K	P���A�#*

logging/current_cost��;��>�+       ��K	����A�#*

logging/current_cost�;E��[+       ��K	n���A�#*

logging/current_cost��;ȁ:+       ��K	L$��A�#*

logging/current_cost��;�6�+       ��K	R��A�#*

logging/current_cost�;9_~T+       ��K	/���A�#*

logging/current_cost�&�;$�U+       ��K	���A�#*

logging/current_cost��;؄��+       ��K	Q���A�#*

logging/current_cost��;D�?�+       ��K	W��A�#*

logging/current_cost�#�;�3�+       ��K	Y8��A�#*

logging/current_cost��;��1�+       ��K	Ff��A�#*

logging/current_cost)0�;�1�+       ��K	)���A�#*

logging/current_cost(�;H�΀+       ��K	#���A�#*

logging/current_cost�G�;T=�+       ��K	����A�#*

logging/current_cost�2�;���+       ��K	!��A�#*

logging/current_costD9�;�ީg+       ��K	MP��A�#*

logging/current_cost'B�;��+       ��K	���A�#*

logging/current_cost O�;�
��+       ��K	����A�#*

logging/current_cost�J�;�6�+       ��K	#���A�#*

logging/current_cost<O�;�'*�+       ��K	 ��A�$*

logging/current_cost�T�;��+       ��K	�7��A�$*

logging/current_cost�J�;
^u�+       ��K	2f��A�$*

logging/current_costK�;�p�y+       ��K	���A�$*

logging/current_costG_�;-�"�+       ��K	����A�$*

logging/current_cost^O�;/��O+       ��K	����A�$*

logging/current_costc�;(��S+       ��K	�"��A�$*

logging/current_cost^�;� ��+       ��K	7Q��A�$*

logging/current_costQ�;̍�A+       ��K	���A�$*

logging/current_cost�X�;ޕK�+       ��K	���A�$*

logging/current_cost�c�;[8�+       ��K	 ���A�$*

logging/current_costnT�;�n�{+       ��K	6��A�$*

logging/current_costc�;@�x�+       ��K	�;��A�$*

logging/current_cost�[�;w�+       ��K	Ah��A�$*

logging/current_cost�_�;���t+       ��K	����A�$*

logging/current_cost4o�;?
+       ��K	����A�$*

logging/current_costS�;@l�Z+       ��K	����A�$*

logging/current_cost�j�;�ɣG+       ��K	_%��A�$*

logging/current_costB_�;>Yd�+       ��K	�T��A�$*

logging/current_costx�;����+       ��K	߃��A�$*

logging/current_costnb�;_-��+       ��K	k���A�$*

logging/current_costbI�;��	s+       ��K	����A�$*

logging/current_costuj�;�85+       ��K	$��A�$*

logging/current_cost�V�;-@3o+       ��K	�C��A�$*

logging/current_cost�U�;Ll�+       ��K	q��A�$*

logging/current_cost�[�;7���+       ��K	5���A�$*

logging/current_cost�`�;��[�+       ��K	u���A�%*

logging/current_cost�X�;��1�+       ��K	����A�%*

logging/current_cost�Y�;��y+       ��K	l-��A�%*

logging/current_cost�^�;���+       ��K	�[��A�%*

logging/current_cost�M�;$_+       ��K	���A�%*

logging/current_costWV�;.�w+       ��K	\���A�%*

logging/current_cost�V�;Oր�+       ��K	����A�%*

logging/current_costd�;ح�+       ��K	���A�%*

logging/current_cost�H�;�A��+       ��K	F��A�%*

logging/current_cost�a�;���+       ��K	�s��A�%*

logging/current_cost�J�;��*�+       ��K	����A�%*

logging/current_cost"h�;{�+       ��K	&���A�%*

logging/current_cost�a�;�u�[+       ��K	7��A�%*

logging/current_costE?�;p+��+       ��K	�0��A�%*

logging/current_costkS�;šr�+       ��K	4_��A�%*

logging/current_cost�J�;��+       ��K	���A�%*

logging/current_costg>�;!�<�+       ��K	���A�%*

logging/current_cost�i�;���m+       ��K	S���A�%*

logging/current_cost�7�;Lmv+       ��K	���A�%*

logging/current_costY�;}
�4+       ��K	*B��A�%*

logging/current_costtT�;�v�5+       ��K	0n��A�%*

logging/current_cost�W�;��T+       ��K	����A�%*

logging/current_cost�7�;&�%�+       ��K	X���A�%*

logging/current_costH�;�|��+       ��K	����A�%*

logging/current_cost7�;R�bG+       ��K	�"��A�%*

logging/current_cost�X�;"�+       ��K	O��A�&*

logging/current_cost4�;���|+       ��K	J~��A�&*

logging/current_cost�Y�;:��h+       ��K	9���A�&*

logging/current_cost�d�;�;��+       ��K	����A�&*

logging/current_cost�8�;�g	+       ��K	'
��A�&*

logging/current_cost�A�;O��5+       ��K	<��A�&*

logging/current_cost�h�;L���+       ��K	�h��A�&*

logging/current_cost��;l�X�+       ��K	����A�&*

logging/current_cost@��;C��k+       ��K	����A�&*

logging/current_cost�;u��:+       ��K	e���A�&*

logging/current_cost�Z�;�\��+       ��K	v��A�&*

logging/current_cost�G�;D\�W+       ��K	�J��A�&*

logging/current_costu<�;��+       ��K	Kx��A�&*

logging/current_costk@�;U�&v+       ��K	���A�&*

logging/current_cost�L�;@�a+       ��K	����A�&*

logging/current_cost�3�;j(#+       ��K	����A�&*

logging/current_cost�J�;���D+       ��K	[*���A�&*

logging/current_cost�j�;]͎�+       ��K	�X���A�&*

logging/current_cost�4�;[�P+       ��K	;����A�&*

logging/current_cost�0�;		>�+       ��K	�����A�&*

logging/current_cost�t�;�߿+       ��K	g����A�&*

logging/current_cost*�;jAK�+       ��K	����A�&*

logging/current_costr;�;=C	�+       ��K	
9���A�&*

logging/current_cost�G�;=Lm+       ��K	,g���A�&*

logging/current_cost�8�;�l�+       ��K	n����A�&*

logging/current_costҀ�;x~f�+       ��K	�����A�&*

logging/current_cost`*�;0DO�+       ��K	�����A�'*

logging/current_cost\/�;���+       ��K	`%���A�'*

logging/current_costYb�;sw��+       ��K	cV���A�'*

logging/current_costy4�;&Iv�+       ��K	�����A�'*

logging/current_costk,�;r��M+       ��K	8����A�'*

logging/current_cost�U�;&�E+       ��K	�����A�'*

logging/current_cost�8�;Z�w�+       ��K	P���A�'*

logging/current_cost�A�;��+       ��K	�:���A�'*

logging/current_cost)1�;�v�+       ��K	�g���A�'*

logging/current_costrZ�;����+       ��K	�����A�'*

logging/current_cost�;�9+       ��K	M����A�'*

logging/current_cost�A�; +       ��K	�����A�'*

logging/current_cost$c�;˸_+       ��K	a$���A�'*

logging/current_cost��;�Gdl+       ��K	<S���A�'*

logging/current_cost9O�;C��B+       ��K	�����A�'*

logging/current_costu.�;7^%�+       ��K	����A�'*

logging/current_cost��;�nA+       ��K	�����A�'*

logging/current_cost�7�;�J�+       ��K	����A�'*

logging/current_cost<E�;KẔ+       ��K	�D���A�'*

logging/current_costb"�;��Hz+       ��K	ls���A�'*

logging/current_costu�;���+       ��K	����A�'*

logging/current_costNK�;��=+       ��K	,����A�'*

logging/current_cost~"�;�]0�+       ��K	D���A�'*

logging/current_cost��;Pjn&+       ��K	�.���A�'*

logging/current_costw!�;``a�+       ��K	�Z���A�'*

logging/current_cost-�;\��+       ��K	�����A�(*

logging/current_cost[$�;�Aqj+       ��K	�����A�(*

logging/current_cost��;���L+       ��K	�����A�(*

logging/current_cost��;Ѧm�+       ��K	,���A�(*

logging/current_cost+D�;|�B�+       ��K	sD���A�(*

logging/current_cost�!�;^Օu+       ��K	�v���A�(*

logging/current_cost�2�;��_+       ��K	�A���A�(*

logging/current_cost�.�;"���+       ��K	Ҽ���A�(*

logging/current_cost�F�;�SA`+       ��K	�3���A�(*

logging/current_cost��;����+       ��K	�t���A�(*

logging/current_cost2,�;)�r+       ��K	9����A�(*

logging/current_cost�;��:L+       ��K	����A�(*

logging/current_cost�%�;D�n+       ��K	4���A�(*

logging/current_cost��;%/d|+       ��K	�O���A�(*

logging/current_cost�*�;�&++       ��K	J����A�(*

logging/current_costB�;C2J�+       ��K	����A�(*

logging/current_costK"�;�I5:+       ��K	U����A�(*

logging/current_costG�;a�0+       ��K	����A�(*

logging/current_cost��;���+       ��K	�K���A�(*

logging/current_cost��;z�'+       ��K	3���A�(*

logging/current_cost��;����+       ��K	�����A�(*

logging/current_cost��;���+       ��K	�����A�(*

logging/current_cost�9�;_�}+       ��K	 ��A�(*

logging/current_cost���;H���+       ��K	q8 ��A�(*

logging/current_cost'��;�D9�+       ��K	�e ��A�(*

logging/current_cost9��;;��+       ��K	�� ��A�(*

logging/current_costd��;��+       ��K	�� ��A�)*

logging/current_cost(�;�5LG+       ��K	� ��A�)*

logging/current_cost;��;T�u�+       ��K	
"��A�)*

logging/current_cost���;�"�D+       ��K	+P��A�)*

logging/current_cost<��;�ռ�+       ��K	�~��A�)*

logging/current_cost�	�;�Q|�+       ��K	V���A�)*

logging/current_cost7��;;���+       ��K	����A�)*

logging/current_cost���;� �+       ��K	�
��A�)*

logging/current_cost���;0@��+       ��K	�?��A�)*

logging/current_cost���;�p+       ��K	�l��A�)*

logging/current_cost���;�2�+       ��K	՟��A�)*

logging/current_costy�;�걺+       ��K	?���A�)*

logging/current_costb��;��+       ��K	c���A�)*

logging/current_costy��;����+       ��K	f&��A�)*

logging/current_costT�;��+       ��K	�Z��A�)*

logging/current_cost���;7FEA+       ��K	(���A�)*

logging/current_cost5��;'O\�+       ��K	,���A�)*

logging/current_cost���;yu-�+       ��K	����A�)*

logging/current_costb�;����+       ��K	���A�)*

logging/current_costY�;��5\+       ��K	�M��A�)*

logging/current_cost���;&�ک+       ��K	{��A�)*

logging/current_cost�$�;G{�+       ��K	ϧ��A�)*

logging/current_cost���;g +       ��K	����A�)*

logging/current_cost���;�#o+       ��K	�
��A�)*

logging/current_cost���;�#Q�+       ��K	�9��A�)*

logging/current_cost��;!��V+       ��K	�g��A�)*

logging/current_cost���;���+       ��K	���A�**

logging/current_costu��;Ǵ�+       ��K	l���A�**

logging/current_cost��;Xc|+       ��K	����A�**

logging/current_cost$��;� ?�+       ��K	�,��A�**

logging/current_cost`��;�g��+       ��K	�_��A�**

logging/current_cost���;��/�+       ��K	ر��A�**

logging/current_cost���;� F�+       ��K	b���A�**

logging/current_cost���;��~�+       ��K	51��A�**

logging/current_cost���;���+       ��K	.p��A�**

logging/current_cost���;\�ja+       ��K	����A�**

logging/current_costu��;���+       ��K	D���A�**

logging/current_costY��;�i��+       ��K	6 ��A�**

logging/current_cost���;C��+       ��K	�Z��A�**

logging/current_cost���;:�?+       ��K	 ���A�**

logging/current_cost���;$}��+       ��K	����A�**

logging/current_cost���;gJ"�+       ��K		��A�**

logging/current_cost��;ֳ��+       ��K	[:	��A�**

logging/current_costd��;���N+       ��K	�m	��A�**

logging/current_cost ��;>6+�+       ��K	z�	��A�**

logging/current_costE��;�/��+       ��K	��	��A�**

logging/current_cost���;�rs�+       ��K	�
��A�**

logging/current_cost��;��
+       ��K	14
��A�**

logging/current_cost���;H*UR+       ��K	(f
��A�**

logging/current_costٌ�;���.+       ��K	�
��A�**

logging/current_costy��;��|�+       ��K	��
��A�**

logging/current_cost5��;X���+       ��K	�
��A�+*

logging/current_cost���;�}�+       ��K	\��A�+*

logging/current_cost9��;�]�+       ��K	EK��A�+*

logging/current_costn��;���Q+       ��K	�z��A�+*

logging/current_cost�z�;9@��+       ��K	���A�+*

logging/current_costR��;�[�+       ��K	���A�+*

logging/current_cost���;
�ۜ+       ��K	{��A�+*

logging/current_cost.��;܏E+       ��K	w6��A�+*

logging/current_costN��;nT�g+       ��K	�h��A�+*

logging/current_cost+��;���9+       ��K	����A�+*

logging/current_cost��;�j�+       ��K	���A�+*

logging/current_cost���;n@��+       ��K	I���A�+*

logging/current_cost��;���+       ��K	�$��A�+*

logging/current_cost���;�e��+       ��K	�^��A�+*

logging/current_cost)��;? cE+       ��K	%���A�+*

logging/current_cost2��;��q+       ��K	ߺ��A�+*

logging/current_cost+��;���+       ��K	����A�+*

logging/current_cost��;� 0+       ��K	���A�+*

logging/current_cost���;�+o+       ��K	�K��A�+*

logging/current_cost���;���+       ��K	~��A�+*

logging/current_cost���;����+       ��K	���A�+*

logging/current_cost�v�;��AL+       ��K	h���A�+*

logging/current_cost���;�4�+       ��K	D��A�+*

logging/current_cost�g�;�� {+       ��K	"1��A�+*

logging/current_costǪ�;L�Ճ+       ��K	/_��A�+*

logging/current_cost���;���+       ��K	n���A�+*

logging/current_cost��;1r�+       ��K	p���A�,*

logging/current_cost�x�;(��+       ��K	d���A�,*

logging/current_cost�h�;�rL9+       ��K	7��A�,*

logging/current_cost���;����+       ��K	�F��A�,*

logging/current_cost|��;� +       ��K	f{��A�,*

logging/current_costpW�;�U��+       ��K	6���A�,*

logging/current_costk/�;�1D+       ��K	F���A�,*

logging/current_cost���;�6��+       ��K	�	��A�,*

logging/current_cost��;�3c�+       ��K	�6��A�,*

logging/current_costyo�;�Vb+       ��K	�c��A�,*

logging/current_cost+��;t�%r+       ��K	����A�,*

logging/current_cost���;W�Wl+       ��K	#���A�,*

logging/current_cost�`�;��(�+       ��K	���A�,*

logging/current_cost�y�;����+       ��K	��A�,*

logging/current_costg��;AtS+       ��K	KL��A�,*

logging/current_cost~I�;�J`+       ��K	�~��A�,*

logging/current_cost���;uW1Y+       ��K	S���A�,*

logging/current_cost ]�;���+       ��K	����A�,*

logging/current_costk��;����+       ��K	���A�,*

logging/current_costm�;�R(+       ��K	�E��A�,*

logging/current_cost�z�;����+       ��K	x��A�,*

logging/current_cost�a�;�y��+       ��K	����A�,*

logging/current_cost�Y�;�"$+       ��K		���A�,*

logging/current_costx�;��+       ��K	���A�,*

logging/current_cost˫�;��K+       ��K	6D��A�,*

logging/current_costUJ�;���U+       ��K	r��A�-*

logging/current_cost��;��+       ��K	k���A�-*

logging/current_cost�b�;�K,E+       ��K	:���A�-*

logging/current_cost�x�;f(+       ��K	���A�-*

logging/current_costː�;�8�t+       ��K	2��A�-*

logging/current_costM�;���$+       ��K	d��A�-*

logging/current_cost���;解�+       ��K	����A�-*

logging/current_cost���;2�=6+       ��K	����A�-*

logging/current_costrT�;��A+       ��K	H���A�-*

logging/current_costi_�;����+       ��K	���A�-*

logging/current_cost�c�;3H �+       ��K	JN��A�-*

logging/current_costw�;[�@v+       ��K	|��A�-*

logging/current_cost0A�;�I+       ��K	����A�-*

logging/current_cost���;�U}+       ��K	����A�-*

logging/current_cost@o�;Rٳ�+       ��K	n��A�-*

logging/current_cost�A�;QB,�+       ��K	�8��A�-*

logging/current_costɒ�;�-��+       ��K	0f��A�-*

logging/current_cost��;h���+       ��K	����A�-*

logging/current_costw��;0'�q+       ��K	*���A�-*

logging/current_costUC�;���u+       ��K	|���A�-*

logging/current_cost��;��+       ��K	�"��A�-*

logging/current_costLG�;��%+       ��K	�O��A�-*

logging/current_cost���;)/��+       ��K	"}��A�-*

logging/current_cost+P�;)W+       ��K	#���A�-*

logging/current_cost�o�;�j�+       ��K	����A�-*

logging/current_costeD�;��r{+       ��K	p��A�-*

logging/current_cost�V�;��L+       ��K	�4��A�.*

logging/current_cost�N�;�c�t+       ��K	�d��A�.*

logging/current_costY��;�E,+       ��K	����A�.*

logging/current_cost���;g���+       ��K	����A�.*

logging/current_cost\H�;�^g&+       ��K	����A�.*

logging/current_cost�~�;v+       ��K	K��A�.*

logging/current_costA�;�(˼+       ��K	�I��A�.*

logging/current_cost�\�;��HC+       ��K	�w��A�.*

logging/current_cost	I�;i��8+       ��K	w���A�.*

logging/current_cost|Z�;Vz�+       ��K	 ���A�.*

logging/current_cost�)�;�%"�+       ��K	� ��A�.*

logging/current_cost�i�;�#ǂ+       ��K	�/��A�.*

logging/current_costuJ�;���+       ��K	�]��A�.*

logging/current_cost)��;���|+       ��K	}���A�.*

logging/current_cost+�;��V/+       ��K	����A�.*

logging/current_costpc�;�Mi(+       ��K	����A�.*

logging/current_cost�x�;�+       ��K	��A�.*

logging/current_cost�B�;"��+       ��K	�E��A�.*

logging/current_cost�m�;�;��+       ��K	s��A�.*

logging/current_cost�N�;A�3m+       ��K	|���A�.*

logging/current_cost8�;��F|+       ��K	����A�.*

logging/current_cost�?�;Zw��+       ��K	���A�.*

logging/current_cost�r�;�B�+       ��K	J-��A�.*

logging/current_cost><�;��?�+       ��K	�^��A�.*

logging/current_cost�S�;���]+       ��K	b���A�.*

logging/current_costE:�;�&#+       ��K	���A�.*

logging/current_costB~�;��%�+       ��K	����A�/*

logging/current_cost"�;��=�+       ��K	���A�/*

logging/current_cost9��;2pg@+       ��K	�E��A�/*

logging/current_cost�U�;�ir
+       ��K	Wr��A�/*

logging/current_cost�H�;��*.+       ��K	���A�/*

logging/current_cost.�;��+       ��K	����A�/*

logging/current_costgo�;gm[+       ��K	����A�/*

logging/current_cost�;x���+       ��K	�,��A�/*

logging/current_costy�;e�t+       ��K	�Y��A�/*

logging/current_cost']�;%� �+       ��K	$���A�/*

logging/current_cost)�;|F�+       ��K	H���A�/*

logging/current_cost5��;S�_�+       ��K	����A�/*

logging/current_costy(�;��0�+       ��K	� ��A�/*

logging/current_costni�;��+       ��K	�G ��A�/*

logging/current_costgR�;|l��+       ��K	0x ��A�/*

logging/current_cost2@�;�#�8+       ��K	�� ��A�/*

logging/current_cost<A�;�\A+       ��K	�� ��A�/*

logging/current_cost.)�;!~�++       ��K	 � ��A�/*

logging/current_cost�^�;n��+       ��K	�-!��A�/*

logging/current_cost� �;���+       ��K	�Y!��A�/*

logging/current_cost���;�#Ǫ+       ��K	̆!��A�/*

logging/current_cost��;�Z+       ��K	w�!��A�/*

logging/current_cost�O�;WU�+       ��K	��!��A�/*

logging/current_costE6�;/��+       ��K	�"��A�/*

logging/current_cost%��;���@+       ��K	�B"��A�/*

logging/current_cost0�;5���+       ��K	�q"��A�0*

logging/current_cost�[�;�u +       ��K	A�"��A�0*

logging/current_cost �;�=o�+       ��K	��"��A�0*

logging/current_cost�2�;Q��+       ��K	v�"��A�0*

logging/current_costpV�;Ԍ��+       ��K	Q%#��A�0*

logging/current_cost�f�;cNC2+       ��K	�V#��A�0*

logging/current_cost� �;�%:n+       ��K	�#��A�0*

logging/current_cost~u�;�y]+       ��K	²#��A�0*

logging/current_cost�%�;^Q`�+       ��K	��#��A�0*

logging/current_costuL�;̚�y+       ��K	�
$��A�0*

logging/current_cost00�;�1�8+       ��K	(:$��A�0*

logging/current_cost�@�;'�.�+       ��K	h$��A�0*

logging/current_cost�5�;�2)+       ��K	m�$��A�0*

logging/current_cost�B�;Iy�q+       ��K	��$��A�0*

logging/current_cost@�;Z�KP+       ��K	��$��A�0*

logging/current_cost�T�;AӇ�+       ��K	�%��A�0*

logging/current_cost�Q�;�em+       ��K		I%��A�0*

logging/current_cost��;��e�+       ��K	-v%��A�0*

logging/current_cost��;�0��+       ��K	ߢ%��A�0*

logging/current_cost[�;�}��+       ��K	��%��A�0*

logging/current_cost9G�;�=�++       ��K	��%��A�0*

logging/current_cost��;.d�+       ��K	0,&��A�0*

logging/current_cost�9�;D=ܢ+       ��K	�Z&��A�0*

logging/current_cost�)�;)b�W+       ��K	��&��A�0*

logging/current_cost�B�;D��+       ��K	b�&��A�0*

logging/current_cost<R�;��	+       ��K	'�&��A�0*

logging/current_coste�;� U+       ��K	+'��A�1*

logging/current_cost{W�;��ԁ+       ��K	�D'��A�1*

logging/current_cost�=�;i�+       ��K	q'��A�1*

logging/current_cost�.�;�n0�+       ��K	f�'��A�1*

logging/current_cost�v�;,���+       ��K	�'��A�1*

logging/current_cost5�;)��+       ��K	C�'��A�1*

logging/current_cost�;�#ml+       ��K	�.(��A�1*

logging/current_costBh�;���+       ��K	�\(��A�1*

logging/current_cost��;��(+       ��K	9�(��A�1*

logging/current_cost�)�;���+       ��K	1�(��A�1*

logging/current_cost\}�;��.w+       ��K	��(��A�1*

logging/current_cost(�;����+       ��K	-)��A�1*

logging/current_cost�6�;���+       ��K	-@)��A�1*

logging/current_cost5]�;�� +       ��K	�k)��A�1*

logging/current_cost�(�;ǟC�+       ��K	��)��A�1*

logging/current_cost0
�;��F�+       ��K	v�)��A�1*

logging/current_cost$Q�;���q+       ��K	��)��A�1*

logging/current_cost"b�;�"g�+       ��K	x+*��A�1*

logging/current_cost���;�S.�+       ��K	Q_*��A�1*

logging/current_cost~�;���+       ��K	B�*��A�1*

logging/current_costy'�;���+       ��K	A�*��A�1*

logging/current_cost;!�;!���+       ��K	��*��A�1*

logging/current_cost$5�;���@+       ��K	;,+��A�1*

logging/current_costNd�;��I+       ��K	�c+��A�1*

logging/current_cost$,�;�S��+       ��K	f�+��A�1*

logging/current_cost7�;̜<+       ��K	��+��A�2*

logging/current_costRa�;o�

+       ��K	0,��A�2*

logging/current_cost�*�;3���+       ��K	3B,��A�2*

logging/current_costy:�;�O�+       ��K	�q,��A�2*

logging/current_cost./�;���V+       ��K	��,��A�2*

logging/current_costND�;���C+       ��K	E�,��A�2*

logging/current_cost\ �;��j+       ��K	x-��A�2*

logging/current_cost�N�;�l�+       ��K	�=-��A�2*

logging/current_cost�J�;�؃8+       ��K	�s-��A�2*

logging/current_cost���;d7�+       ��K	��-��A�2*

logging/current_costT�;���+       ��K	Q�-��A�2*

logging/current_cost�R�;�t��+       ��K	.��A�2*

logging/current_cost�;�#[I+       ��K	D.��A�2*

logging/current_cost��;��N+       ��K	Bw.��A�2*

logging/current_costo�;�>��+       ��K	Ч.��A�2*

logging/current_costg�;��f1+       ��K	[�.��A�2*

logging/current_cost�'�;�2��+       ��K	/��A�2*

logging/current_cost�;�;$^�+       ��K	�>/��A�2*

logging/current_costNB�;�KT�+       ��K	�t/��A�2*

logging/current_cost ,�;M�f~+       ��K	g�/��A�2*

logging/current_cost�,�;}� +       ��K	A�/��A�2*

logging/current_cost�#�;X���+       ��K	 0��A�2*

logging/current_cost�]�;�1��+       ��K	�/0��A�2*

logging/current_costE�;N��+       ��K	�^0��A�2*

logging/current_costY-�;�̤^+       ��K	ǌ0��A�2*

logging/current_cost5G�;
y�+       ��K	��0��A�2*

logging/current_cost2=�;z��+       ��K	��0��A�3*

logging/current_cost\�;���0+       ��K	�1��A�3*

logging/current_cost>F�;�N&+       ��K	fC1��A�3*

logging/current_cost��;+J�P+       ��K	�p1��A�3*

logging/current_cost��;��+       ��K	�1��A�3*

logging/current_costQ�;���}+       ��K	��1��A�3*

logging/current_costUW�; �Y+       ��K	��1��A�3*

logging/current_cost��;c)�+       ��K	'2��A�3*

logging/current_cost�	�;m`��+       ��K	�U2��A�3*

logging/current_cost,�;W��+       ��K	9�2��A�3*

logging/current_cost�N�;\*�+       ��K	��2��A�3*

logging/current_cost��;1��+       ��K	��2��A�3*

logging/current_cost��;AV��+       ��K	�3��A�3*

logging/current_cost�H�;/p�(+       ��K	�53��A�3*

logging/current_cost�@�;���+       ��K	�c3��A�3*

logging/current_cost��;E�:+       ��K	��3��A�3*

logging/current_costg�;ۥ+       ��K	'�3��A�3*

logging/current_costR7�;J[�Y+       ��K	��3��A�3*

logging/current_cost�"�;�%�Y+       ��K	�4��A�3*

logging/current_cost�;c�+       ��K	G4��A�3*

logging/current_cost�@�;�(B�+       ��K	s4��A�3*

logging/current_costR�;�tS�+       ��K	��4��A�3*

logging/current_costu7�;b�X+       ��K	(�4��A�3*

logging/current_cost���;"r�+       ��K	�4��A�3*

logging/current_cost�_�;;��4+       ��K	+*5��A�3*

logging/current_costgR�;�iT�+       ��K	�V5��A�3*

logging/current_cost 9�;�{��+       ��K	��5��A�4*

logging/current_cost��;no^+       ��K	ۯ5��A�4*

logging/current_cost��;���+       ��K	h�5��A�4*

logging/current_cost�s�;�K��+       ��K	�	6��A�4*

logging/current_cost %�;s-�+       ��K	n56��A�4*

logging/current_cost.�;d�Ͽ+       ��K	?b6��A�4*

logging/current_costt�;
���+       ��K	Տ6��A�4*

logging/current_cost+.�;��"+       ��K	��6��A�4*

logging/current_cost|-�;7��;+       ��K	��6��A�4*

logging/current_cost�*�;�x!+       ��K	7��A�4*

logging/current_cost5��;�� 7+       ��K	xD7��A�4*

logging/current_cost9Q�;aF��+       ��K	=r7��A�4*

logging/current_costf�;�5��+       ��K	7�7��A�4*

logging/current_cost��;�D&�+       ��K	U�7��A�4*

logging/current_cost�B�;��N�+       ��K	/8��A�4*

logging/current_cost��;��+       ��K	438��A�4*

logging/current_costg��;��xJ+       ��K	Ab8��A�4*

logging/current_cost��;N�r+       ��K	��8��A�4*

logging/current_costU�;�ZϨ+       ��K	��8��A�4*

logging/current_cost�'�;K�{�+       ��K	i�8��A�4*

logging/current_cost M�;ܹ�	+       ��K	�"9��A�4*

logging/current_cost��;iq+       ��K	�S9��A�4*

logging/current_cost{;�;��_+       ��K	��9��A�4*

logging/current_cost��;����+       ��K	ï9��A�4*

logging/current_cost�k�;N1��+       ��K	"�9��A�4*

logging/current_cost��;)T�+       ��K	:��A�5*

logging/current_cost9C�;KGyl+       ��K	b?:��A�5*

logging/current_cost��;KzШ+       ��K	5l:��A�5*

logging/current_cost�G�;&���+       ��K	��:��A�5*

logging/current_costGD�;'�ڽ+       ��K	�:��A�5*

logging/current_cost)��;5�;+       ��K	P�:��A�5*

logging/current_cost~��;�+       ��K	);��A�5*

logging/current_cost�-�;9���+       ��K	�\;��A�5*

logging/current_cost�|�;Q�7�+       ��K	��;��A�5*

logging/current_cost9��;����+       ��K	!�;��A�5*

logging/current_cost\�;5qgQ+       ��K	�<��A�5*

logging/current_cost��;��0�+       ��K	!P<��A�5*

logging/current_costiT�;2�;E+       ��K	Ġ<��A�5*

logging/current_cost��;��+       ��K	��<��A�5*

logging/current_cost\��;<M��+       ��K	L	=��A�5*

logging/current_costD�;�f�#+       ��K	QF=��A�5*

logging/current_cost��;5,�+       ��K	��=��A�5*

logging/current_costw�;���+       ��K	��=��A�5*

logging/current_cost���;V$�+       ��K	+>��A�5*

logging/current_cost���;�]��+       ��K	`=>��A�5*

logging/current_costdW�;��[+       ��K	1r>��A�5*

logging/current_costI<�;X��+       ��K	��>��A�5*

logging/current_cost$#�;���+       ��K	,�>��A�5*

logging/current_cost��;�0�+       ��K	�?��A�5*

logging/current_cost���;aG3+       ��K	�H?��A�5*

logging/current_cost�>�;�FX�+       ��K	�?��A�5*

logging/current_cost���;W��+       ��K	�?��A�6*

logging/current_cost���; l*�+       ��K	��?��A�6*

logging/current_cost�;��+       ��K	f@��A�6*

logging/current_costP��;����+       ��K	�U@��A�6*

logging/current_cost�J�;�+l�+       ��K	��@��A�6*

logging/current_cost�T�;��S+       ��K	Y�@��A�6*

logging/current_cost��;Dc�+       ��K	�A��A�6*

logging/current_cost��;���h+       ��K	�FA��A�6*

logging/current_cost���;{B+       ��K	�A��A�6*

logging/current_cost�/�;\`j�+       ��K	��A��A�6*

logging/current_cost�'�;g~G�+       ��K	�A��A�6*

logging/current_cost2�;��+       ��K	3"B��A�6*

logging/current_costy��;
��+       ��K	�^B��A�6*

logging/current_cost2F�;�݁+       ��K	��B��A�6*

logging/current_cost�"�;��U+       ��K	)�B��A�6*

logging/current_cost��;'(6+       ��K	��B��A�6*

logging/current_cost���;#À�+       ��K	4=C��A�6*

logging/current_costK��;H�j�+       ��K	$vC��A�6*

logging/current_cost5O�;W��+       ��K	��C��A�6*

logging/current_cost�=�;]��+       ��K	��C��A�6*

logging/current_cost��;*|u+       ��K	/D��A�6*

logging/current_cost��;Q�i�+       ��K	^CD��A�6*

logging/current_costd�;8{h9+       ��K	<uD��A�6*

logging/current_cost,A�;��F9+       ��K	��D��A�6*

logging/current_cost��;4�a+       ��K	��D��A�6*

logging/current_cost���;����+       ��K	�E��A�7*

logging/current_costY��;
��+       ��K	�3E��A�7*

logging/current_costn�;w��+       ��K	heE��A�7*

logging/current_cost_�;(}Ү+       ��K	D�E��A�7*

logging/current_costD��;Z6s+       ��K	��E��A�7*

logging/current_cost���;�&ֳ+       ��K	4�E��A�7*

logging/current_cost���;?�ɬ+       ��K		-F��A�7*

logging/current_costEu�;⿧R+       ��K	�eF��A�7*

logging/current_cost�"�;��R�+       ��K	��F��A�7*

logging/current_costP��;�[�+       ��K	!�F��A�7*

logging/current_cost�:�;���+       ��K	7 G��A�7*

logging/current_cost���;W���+       ��K	�.G��A�7*

logging/current_cost��;ކ��+       ��K	HbG��A�7*

logging/current_cost�O�;��+       ��K	��G��A�7*

logging/current_cost~>�;�� �+       ��K	��G��A�7*

logging/current_costG%�;�gH+       ��K	��G��A�7*

logging/current_cost^�;yn��+       ��K	#H��A�7*

logging/current_cost.(�;��0�+       ��K	�SH��A�7*

logging/current_cost���;�qU+       ��K	�H��A�7*

logging/current_cost9�;�Cn+       ��K	M�H��A�7*

logging/current_costi/�;�N�+       ��K	6�H��A�7*

logging/current_cost�%�;l#ճ+       ��K	
I��A�7*

logging/current_cost\}�;S8�+       ��K	�BI��A�7*

logging/current_cost�v�;j��+       ��K	}qI��A�7*

logging/current_cost��;�*<�+       ��K	r�I��A�7*

logging/current_cost; �;+1>+       ��K	��I��A�7*

logging/current_cost�z�;4���+       ��K	��I��A�8*

logging/current_cost�X�;hS��+       ��K	r2J��A�8*

logging/current_cost Y�;P�f�+       ��K	�bJ��A�8*

logging/current_cost��;�u�+       ��K	M�J��A�8*

logging/current_cost�,�;MD��+       ��K	�J��A�8*

logging/current_cost�_�;ư�+       ��K	��J��A�8*

logging/current_cost\��;O�+       ��K	K��A�8*

logging/current_cost@g�;*�Ď+       ��K	cQK��A�8*

logging/current_costR��;�y@+       ��K	P�K��A�8*

logging/current_cost�N�;��f+       ��K	A�K��A�8*

logging/current_cost�a�;<��+       ��K	n�K��A�8*

logging/current_cost���;|�6�+       ��K	�L��A�8*

logging/current_cost�`�;RB�g+       ��K	�3L��A�8*

logging/current_cost���;4Rb�+       ��K	�fL��A�8*

logging/current_costdC�;.@�>+       ��K	�L��A�8*

logging/current_cost��;���:+       ��K	��L��A�8*

logging/current_cost�f�;G��+       ��K	{�L��A�8*

logging/current_cost���;���q+       ��K	Z%M��A�8*

logging/current_costKa�;J�ԅ+       ��K	�XM��A�8*

logging/current_cost�(�;Ҭr�+       ��K	y�M��A�8*

logging/current_cost�n�;��w+       ��K	r�M��A�8*

logging/current_cost���;8��+       ��K	��M��A�8*

logging/current_cost+M�;�J++       ��K	ON��A�8*

logging/current_cost"�;��o+       ��K	AN��A�8*

logging/current_cost..�;����+       ��K	uN��A�8*

logging/current_cost�K�;��LF+       ��K	V�N��A�8*

logging/current_cost���;�A�N+       ��K	��N��A�9*

logging/current_cost��;M֘�+       ��K	��N��A�9*

logging/current_cost�1�;�`��+       ��K	;+O��A�9*

logging/current_cost@�;�/+       ��K	�YO��A�9*

logging/current_costy�;���q+       ��K	��O��A�9*

logging/current_cost�a�;�r�+       ��K	��O��A�9*

logging/current_cost"��;��!+       ��K	��O��A�9*

logging/current_cost��;}-�+       ��K	�P��A�9*

logging/current_costt��;�PB�+       ��K	�HP��A�9*

logging/current_costUK�;}���+       ��K	+vP��A�9*

logging/current_cost|0�;͒�[+       ��K	?�P��A�9*

logging/current_costν�;OO�+       ��K	�P��A�9*

logging/current_cost���;�%+       ��K	�Q��A�9*

logging/current_cost5��;�z�]+       ��K	75Q��A�9*

logging/current_cost,.�;���+       ��K	!fQ��A�9*

logging/current_cost'A�;����+       ��K	r�Q��A�9*

logging/current_cost@��;�wd�+       ��K	��Q��A�9*

logging/current_costl�;�	��+       ��K	��Q��A�9*

logging/current_costd��;����+       ��K	�#R��A�9*

logging/current_cost��;��+       ��K	WR��A�9*

logging/current_costR.�;��)+       ��K	�R��A�9*

logging/current_cost���;�o�3+       ��K	۳R��A�9*

logging/current_costz�;oǩ�+       ��K	��R��A�9*

logging/current_cost���;35+       ��K	-S��A�9*

logging/current_cost���;��E+       ��K	BS��A�9*

logging/current_cost��;��o+       ��K	�oS��A�:*

logging/current_cost9��;$2$+       ��K	n�S��A�:*

logging/current_cost�;Wu�3+       ��K	��S��A�:*

logging/current_cost�$�;�-�}+       ��K	T��A�:*

logging/current_cost���;��#�+       ��K	b6T��A�:*

logging/current_costt��; ��+       ��K	4eT��A�:*

logging/current_cost9_�;��]+       ��K	�T��A�:*

logging/current_cost��;��+       ��K	�T��A�:*

logging/current_cost���;�>�+       ��K	9�T��A�:*

logging/current_costBW�;�¯+       ��K	�U��A�:*

logging/current_cost���;�Ur+       ��K	�HU��A�:*

logging/current_cost+!�;���>+       ��K	IvU��A�:*

logging/current_cost'��;#�f+       ��K	+�U��A�:*

logging/current_cost.��;�"�+       ��K	�U��A�:*

logging/current_cost�F�;�X�S+       ��K	��U��A�:*

logging/current_cost�Q�;E��+       ��K	G0V��A�:*

logging/current_cost���;KdMJ+       ��K	6]V��A�:*

logging/current_cost%��;�x3b+       ��K	��V��A�:*

logging/current_costke�;Wa�+       ��K	 �V��A�:*

logging/current_cost�>�;Z��F+       ��K	W�V��A�:*

logging/current_cost�4�;��[�+       ��K	#W��A�:*

logging/current_costΩ�;h��+       ��K	�BW��A�:*

logging/current_cost5�;Ꭽ)+       ��K	avW��A�:*

logging/current_cost��;�GD\+       ��K	ߣW��A�:*

logging/current_cost��;5��X+       ��K	�W��A�:*

logging/current_cost�@�;���+       ��K	� X��A�:*

logging/current_cost���;��Z+       ��K	0X��A�;*

logging/current_cost��;3�;+       ��K	�^X��A�;*

logging/current_cost9I�;���+       ��K	�X��A�;*

logging/current_costT�;⦟+       ��K	&�X��A�;*

logging/current_cost�P�;�6v�+       ��K	��X��A�;*

logging/current_cost���;��^�+       ��K	�Y��A�;*

logging/current_cost�h�;C��T+       ��K	rHY��A�;*

logging/current_cost��;Ӟ�+       ��K	+xY��A�;*

logging/current_costg7�;<�+       ��K	��Y��A�;*

logging/current_costŦ�;��/�+       ��K	��Y��A�;*

logging/current_cost��;��S[+       ��K	Z��A�;*

logging/current_cost���;���x+       ��K	�/Z��A�;*

logging/current_costGZ�;��dB+       ��K	�\Z��A�;*

logging/current_cost���;0ٜ;+       ��K	��Z��A�;*

logging/current_costd��;9B�+       ��K	˹Z��A�;*

logging/current_costD��;`8V+       ��K	��Z��A�;*

logging/current_cost�T�;h��+       ��K	)[��A�;*

logging/current_cost5��;}=+       ��K	�@[��A�;*

logging/current_cost���;�yp{+       ��K	�l[��A�;*

logging/current_cost��;���+       ��K	ۗ[��A�;*

logging/current_cost���;S�LJ+       ��K	�[��A�;*

logging/current_costLp�;���R+       ��K	��[��A�;*

logging/current_cost���;p�+       ��K	�%\��A�;*

logging/current_costk��;?1�+       ��K	DR\��A�;*

logging/current_cost'��;~F�+       ��K	&\��A�;*

logging/current_cost"�;��'+       ��K	ޫ\��A�<*

logging/current_costt�;)�+       ��K	�\��A�<*

logging/current_cost4��;�J�+       ��K	�]��A�<*

logging/current_cost��;Dʤ=+       ��K	5]��A�<*

logging/current_costd��;�%��+       ��K	+b]��A�<*

logging/current_cost$]�;d�!-+       ��K	��]��A�<*

logging/current_cost��;D�7o+       ��K	��]��A�<*

logging/current_cost���;�e=-+       ��K	��]��A�<*

logging/current_costD��;�$�+       ��K	�^��A�<*

logging/current_cost��;]��{+       ��K	�@^��A�<*

logging/current_cost�L�;�-�+       ��K	%r^��A�<*

logging/current_costB"�;�[�+       ��K	��^��A�<*

logging/current_cost���;�1�+       ��K	��^��A�<*

logging/current_cost���;pxfw+       ��K	R_��A�<*

logging/current_cost���;W9z+       ��K	�-_��A�<*

logging/current_costyU�;ۯ-�+       ��K	.\_��A�<*

logging/current_cost���;���+       ��K	͈_��A�<*

logging/current_cost���;�++       ��K	N�_��A�<*

logging/current_cost���;�@pb+       ��K	B�_��A�<*

logging/current_cost�,�;M.N+       ��K	(`��A�<*

logging/current_costBf�;�E�6+       ��K	NC`��A�<*

logging/current_cost~��;C7+       ��K	pp`��A�<*

logging/current_cost.��;.bez+       ��K	#�`��A�<*

logging/current_costy��;i	+       ��K	A�`��A�<*

logging/current_cost0�;�ES+       ��K	��`��A�<*

logging/current_cost+>�;!W�+       ��K	�-a��A�<*

logging/current_cost���;b|��+       ��K	~]a��A�=*

logging/current_cost���;C\�J+       ��K	��a��A�=*

logging/current_cost���;�.�L+       ��K	H�a��A�=*

logging/current_cost�<�;����+       ��K	��a��A�=*

logging/current_costU>�;]�W+       ��K	�b��A�=*

logging/current_costDv�; �c+       ��K	Jb��A�=*

logging/current_costg��;WU3h+       ��K	Hwb��A�=*

logging/current_cost��;\�Y�+       ��K	[�b��A�=*

logging/current_cost��;g���+       ��K	*�b��A�=*

logging/current_cost$'�;��+       ��K	� c��A�=*

logging/current_cost`��;�/0+       ��K	�-c��A�=*

logging/current_costd��;'��7+       ��K	�Zc��A�=*

logging/current_cost��;?��/+       ��K	U�c��A�=*

logging/current_cost��;�泽+       ��K	l�c��A�=*

logging/current_coste�;����+       ��K	��c��A�=*

logging/current_cost4��;��.+       ��K	d��A�=*

logging/current_cost�w�;t6��+       ��K	tKd��A�=*

logging/current_cost��;z�.$+       ��K	�wd��A�=*

logging/current_cost�q�;��*+       ��K	��d��A�=*

logging/current_cost4s�;Th+       ��K	��d��A�=*

logging/current_costd�;?@�+       ��K	� e��A�=*

logging/current_cost���;6Ms�+       ��K	�-e��A�=*

logging/current_cost�(�;��ۈ+       ��K	i[e��A�=*

logging/current_costK��;ၢ�+       ��K	��e��A�=*

logging/current_cost f�;�y�+       ��K	�e��A�=*

logging/current_cost���;��
z+       ��K	��e��A�=*

logging/current_costD��;쒸�+       ��K	�f��A�>*

logging/current_cost�F�;�s�+       ��K	�?f��A�>*

logging/current_costT�;f�+       ��K	]nf��A�>*

logging/current_cost���;��+       ��K	�f��A�>*

logging/current_costt�;Á�+       ��K	��f��A�>*

logging/current_costr��;k_x
+       ��K	�f��A�>*

logging/current_cost^;�;|Z�+       ��K	�+g��A�>*

logging/current_costD(�;��3+       ��K	�Xg��A�>*

logging/current_cost|N�;���X+       ��K	Y�g��A�>*

logging/current_cost��;eK�+       ��K	e�g��A�>*

logging/current_cost���;Ъ�+       ��K	��g��A�>*

logging/current_costI�;��+       ��K	�h��A�>*

logging/current_costNq�;��Z'+       ��K	^?h��A�>*

logging/current_cost��;~Y�+       ��K	lh��A�>*

logging/current_cost'��;��Q+       ��K	}�h��A�>*

logging/current_cost���;�CT�+       ��K	}�h��A�>*

logging/current_costU�;�!��+       ��K	��h��A�>*

logging/current_costw��;��3+       ��K	�!i��A�>*

logging/current_cost�l�;� �+       ��K	�Mi��A�>*

logging/current_costPz�;��a+       ��K	Z{i��A�>*

logging/current_cost��;v2T�+       ��K	�i��A�>*

logging/current_cost�C�;�72�+       ��K	��i��A�>*

logging/current_cost���;�Ow;+       ��K	j��A�>*

logging/current_cost��;w�hc+       ��K	�:j��A�>*

logging/current_costuD�;��K=+       ��K	%gj��A�>*

logging/current_cost���;D| +       ��K	�j��A�?*

logging/current_costd�;)�^�+       ��K	��j��A�?*

logging/current_cost���;�h+       ��K	"�j��A�?*

logging/current_cost�I�;��+       ��K	k��A�?*

logging/current_cost,G�;Ψ.0+       ��K	;Hk��A�?*

logging/current_cost���;[J�q+       ��K	�vk��A�?*

logging/current_cost�1�;�+��+       ��K	��k��A�?*

logging/current_cost��;�ֻ+       ��K	��k��A�?*

logging/current_costP,�;H�bm+       ��K	�l��A�?*

logging/current_cost+��;����+       ��K	O>l��A�?*

logging/current_costK��;4<�o+       ��K	�ml��A�?*

logging/current_cost��;V���+       ��K	��l��A�?*

logging/current_cost��;b~��+       ��K	��l��A�?*

logging/current_costK��;Nt�q+       ��K	��l��A�?*

logging/current_costNl�;���+       ��K	(m��A�?*

logging/current_costk�;��+       ��K	�Xm��A�?*

logging/current_costLA�;LI��+       ��K	��m��A�?*

logging/current_cost9��;�'��+       ��K	
�m��A�?*

logging/current_cost@_�;�z<�+       ��K	��m��A�?*

logging/current_cost��;0:�+       ��K	)n��A�?*

logging/current_costY��;S}6h+       ��K	�An��A�?*

logging/current_costrW�;��1�+       ��K	�pn��A�?*

logging/current_cost�;ȃI�+       ��K	n�n��A�?*

logging/current_cost�Z�;P�3�+       ��K	��n��A�?*

logging/current_costU��;�t�+       ��K	�o��A�?*

logging/current_cost��;)/	�+       ��K	�/o��A�?*

logging/current_cost4<�;���+       ��K	�]o��A�@*

logging/current_cost�f�;~s�_+       ��K	��o��A�@*

logging/current_cost�c�;�6�+       ��K	<�o��A�@*

logging/current_cost�J�;;|�+       ��K	G�o��A�@*

logging/current_cost0�;� �q+       ��K	�p��A�@*

logging/current_cost��;��5+       ��K	�Jp��A�@*

logging/current_cost�U�;�m;�+       ��K	3{p��A�@*

logging/current_cost�o�;�p$+       ��K	��p��A�@*

logging/current_cost'��;bfM+       ��K	��p��A�@*

logging/current_cost8�;F%8�+       ��K	\q��A�@*

logging/current_cost�;�+�E+       ��K	�3q��A�@*

logging/current_costJ�;�"�+       ��K	bq��A�@*

logging/current_costd^�;w�k+       ��K	��q��A�@*

logging/current_costb�;0� +       ��K	��q��A�@*

logging/current_cost=�;A{5�+       ��K	`�q��A�@*

logging/current_cost�!�;�G��+       ��K	�r��A�@*

logging/current_cost�1�;���+       ��K	�Ir��A�@*

logging/current_cost���;|ćO+       ��K	Nvr��A�@*

logging/current_cost���;O]
�+       ��K	h�r��A�@*

logging/current_cost!�;&�G+       ��K	��r��A�@*

logging/current_cost)��;}M@�+       ��K	��r��A�@*

logging/current_cost�N�;~�F+       ��K	&s��A�@*

logging/current_costgJ�;��@+       ��K	OSs��A�@*

logging/current_cost	��;'�a+       ��K	��s��A�@*

logging/current_costU�;P���+       ��K	��s��A�@*

logging/current_cost���;�%B�+       ��K	H�s��A�A*

logging/current_cost���;j:�+       ��K	�t��A�A*

logging/current_cost��;ꥫ:+       ��K	:t��A�A*

logging/current_costG��;E<3W+       ��K	�ht��A�A*

logging/current_cost,W�;�x�+       ��K	֖t��A�A*

logging/current_cost���;	��4+       ��K	�t��A�A*

logging/current_cost���;���+       ��K	r�t��A�A*

logging/current_costD�;Y5H�+       ��K	�u��A�A*

logging/current_cost���;��f�+       ��K	�Lu��A�A*

logging/current_costr��;��+       ��K	{u��A�A*

logging/current_cost���;j^E+       ��K	ܦu��A�A*

logging/current_cost���;`hN+       ��K	��u��A�A*

logging/current_cost�h�;�Kh�+       ��K	�v��A�A*

logging/current_cost��;%ȱP+       ��K	�4v��A�A*

logging/current_costk��;Q-l�+       ��K	�bv��A�A*

logging/current_cost�; ��+       ��K	!�v��A�A*

logging/current_cost�$�;R�H+       ��K	׽v��A�A*

logging/current_cost�`�;Y8C#+       ��K	��v��A�A*

logging/current_cost.��;5��+       ��K	�w��A�A*

logging/current_cost�2�;����+       ��K	�Gw��A�A*

logging/current_cost���;�,��+       ��K	�tw��A�A*

logging/current_cost�[�;�+       ��K	i�w��A�A*

logging/current_cost�2�;�*�#+       ��K	��w��A�A*

logging/current_cost���;��(+       ��K	�w��A�A*

logging/current_cost��;}xI�+       ��K	�)x��A�A*

logging/current_cost�s�;De��+       ��K	�Vx��A�A*

logging/current_costU��;X�U+       ��K	��x��A�B*

logging/current_cost��;S���+       ��K	��x��A�B*

logging/current_cost���;�,�?+       ��K	��x��A�B*

logging/current_cost$k�;�צf+       ��K	�y��A�B*

logging/current_costd��;f0LX+       ��K	�Fy��A�B*

logging/current_costin�;�~�x+       ��K	ity��A�B*

logging/current_cost�m�;��1+       ��K	�y��A�B*

logging/current_costk��;pt��+       ��K	r�y��A�B*

logging/current_costd��;�\A+       ��K	^z��A�B*

logging/current_cost¬�;VE�/+       ��K	|.z��A�B*

logging/current_cost���;vϑ+       ��K	�az��A�B*

logging/current_costuU�;��i+       ��K	Ǝz��A�B*

logging/current_costTN�;�s�3+       ��K	Z�z��A�B*

logging/current_cost7-�;�%��+       ��K	d�z��A�B*

logging/current_cost\�;�:��+       ��K	�{��A�B*

logging/current_costR��;���+       ��K	�J{��A�B*

logging/current_cost)@�;H�TL+       ��K	|�{��A�B*

logging/current_costR��;U�'+       ��K	_�{��A�B*

logging/current_costu��;�^��+       ��K	j |��A�B*

logging/current_cost$�;[���+       ��K	_.|��A�B*

logging/current_costŃ�;_A�u+       ��K	![|��A�B*

logging/current_costҶ�;���+       ��K	o�|��A�B*

logging/current_cost���;|OF+       ��K	k�|��A�B*

logging/current_costr�;��_+       ��K	U�|��A�B*

logging/current_cost��;_~n�+       ��K	�}��A�B*

logging/current_costP��;�k[
+       ��K	JK}��A�B*

logging/current_cost@�;��%+       ��K	�z}��A�C*

logging/current_cost���;)�-+       ��K	b�}��A�C*

logging/current_cost��;8�R+       ��K	��}��A�C*

logging/current_costbF�;���n+       ��K	�~��A�C*

logging/current_cost5��;�M�|+       ��K	/3~��A�C*

logging/current_cost�;~h��+       ��K	�`~��A�C*

logging/current_cost��;�j{
+       ��K	�~��A�C*

logging/current_cost2��;4R��+       ��K	��~��A�C*

logging/current_cost�%�;w��N+       ��K	��~��A�C*

logging/current_cost�N�;�-�+       ��K	���A�C*

logging/current_costЪ�;�s�+       ��K	�J��A�C*

logging/current_cost���;F9�,+       ��K	�y��A�C*

logging/current_cost��;	�d+       ��K	ħ��A�C*

logging/current_cost�_�;�ð�+       ��K	����A�C*

logging/current_costy��;�TJ+       ��K	2���A�C*

logging/current_cost���;
�Z+       ��K	�1���A�C*

logging/current_cost{��;`=�W+       ��K	�_���A�C*

logging/current_cost��;�	��+       ��K	�����A�C*

logging/current_cost���;y��+       ��K	����A�C*

logging/current_cost���;����+       ��K	/쀔�A�C*

logging/current_costdt�;#o-�+       ��K	����A�C*

logging/current_cost��;��1+       ��K	cL���A�C*

logging/current_cost+a�;��+       ��K	�y���A�C*

logging/current_cost���;��G+       ��K	�����A�C*

logging/current_coste��;j�P�+       ��K	�ׁ��A�C*

logging/current_cost�,�;��e�+       ��K	x���A�D*

logging/current_cost���;!>o5+       ��K	~4���A�D*

logging/current_costY��;���+       ��K	�a���A�D*

logging/current_costG��;w�+       ��K	#����A�D*

logging/current_cost�Y�;��,+       ��K	�����A�D*

logging/current_cost\��;�Z�+       ��K	��A�D*

logging/current_costr��;��i�+       ��K	_���A�D*

logging/current_cost��;�*\�+       ��K	OJ���A�D*

logging/current_cost���;+�q+       ��K	�y���A�D*

logging/current_cost[l�;r+       ��K	�����A�D*

logging/current_costS�;o7��+       ��K	y׃��A�D*

logging/current_cost&�;����+       ��K	����A�D*

logging/current_cost��;�ɳ9+       ��K	2���A�D*

logging/current_costU�;7G܄+       ��K	�i���A�D*

logging/current_cost5l�;G�s�+       ��K	b����A�D*

logging/current_cost���;�ͣ+       ��K	`��A�D*

logging/current_cost'A�;X�BS+       ��K	E����A�D*

logging/current_costUg�;}^�+       ��K	����A�D*

logging/current_cost.��;b`�+       ��K	�K���A�D*

logging/current_costKP�;Tj+       ��K	x���A�D*

logging/current_cost��;C�+       ��K	����A�D*

logging/current_cost9��;&��+       ��K	]݅��A�D*

logging/current_cost@��;��-"+       ��K	|���A�D*

logging/current_costn�;@~�4+       ��K	E���A�D*

logging/current_cost�m�;�e+�+       ��K	v���A�D*

logging/current_cost>��;Cl2+       ��K	����A�D*

logging/current_cost7�;�+CI+       ��K	@ֆ��A�E*

logging/current_cost���;2%�m+       ��K	����A�E*

logging/current_cost �;��9�+       ��K	7���A�E*

logging/current_cost ��;�z�+       ��K	�c���A�E*

logging/current_cost;��;�++       ��K	a����A�E*

logging/current_cost��;�羪+       ��K	GǇ��A�E*

logging/current_costU��;���+       ��K	0��A�E*

logging/current_cost9��;_$�++       ��K	�#���A�E*

logging/current_cost|�;�~��+       ��K	�S���A�E*

logging/current_cost�K�;���/+       ��K	�����A�E*

logging/current_cost>��;��US+       ��K	����A�E*

logging/current_cost��;���+       ��K	rވ��A�E*

logging/current_cost�^�;ۘ�+       ��K	
���A�E*

logging/current_cost���;/��`+       ��K	X@���A�E*

logging/current_cost���;��{+       ��K	�n���A�E*

logging/current_cost%��;�;m+       ��K	n����A�E*

logging/current_cost�6�;��z+       ��K	;Љ��A�E*

logging/current_costN��;��o
+       ��K	9����A�E*

logging/current_cost�T�;�T�h+       ��K	,���A�E*

logging/current_cost��;���+       ��K	8Y���A�E*

logging/current_cost<
�;�9Z+       ��K	J����A�E*

logging/current_cost_�;�y�+       ��K	�����A�E*

logging/current_cost�-�;�RW�+       ��K	抔�A�E*

logging/current_cost���;<�+       ��K	���A�E*

logging/current_cost�:�;�_�|+       ��K	�F���A�E*

logging/current_costT��;�^�Q+       ��K	�u���A�F*

logging/current_costeu�;���+       ��K	󣋔�A�F*

logging/current_costY��;��x{+       ��K	|؋��A�F*

logging/current_cost�'�;���C+       ��K	����A�F*

logging/current_cost9��;Ud�+       ��K	N5���A�F*

logging/current_cost�z�;<��+       ��K	�b���A�F*

logging/current_cost4��;݋��+       ��K	>����A�F*

logging/current_cost� �;t�+       ��K	@����A�F*

logging/current_costM�;ŭ[B+       ��K	n댔�A�F*

logging/current_costy�;�f�+       ��K	�?���A�F*

logging/current_cost���;D��+       ��K	{���A�F*

logging/current_cost;H�;���6+       ��K	j����A�F*

logging/current_cost>��;���+       ��K	>
���A�F*

logging/current_costE��; /�P+       ��K	�T���A�F*

logging/current_costҞ�;��=+       ��K	J����A�F*

logging/current_cost�'�;�X�
+       ��K	�Ύ��A�F*

logging/current_costr��;؊65+       ��K	����A�F*

logging/current_cost2%�;��=+       ��K	R=���A�F*

logging/current_cost���;����+       ��K	􂏔�A�F*

logging/current_cost�Q�;�h]+       ��K	A����A�F*

logging/current_cost� �;m���+       ��K	{Ꮤ�A�F*

logging/current_cost���;���Q+       ��K	����A�F*

logging/current_cost�p�;c��+       ��K	V���A�F*

logging/current_cost�?�;��:1+       ��K	�����A�F*

logging/current_costҧ�;1���+       ��K		����A�F*

logging/current_costU3�;�Bf}+       ��K	���A�F*

logging/current_cost���;B��x+       ��K	0���A�G*

logging/current_cost4��;�R y+       ��K	qg���A�G*

logging/current_cost��;���+       ��K	㚑��A�G*

logging/current_cost�]�;����+       ��K	�֑��A�G*

logging/current_costN��;�N��+       ��K	����A�G*

logging/current_cost���;z��|+       ��K	]B���A�G*

logging/current_costK��;0���+       ��K	zr���A�G*

logging/current_cost��;�cĎ+       ��K	�����A�G*

logging/current_cost�E�;;�$+       ��K	NӒ��A�G*

logging/current_cost��;l��+       ��K	2���A�G*

logging/current_cost��;��+       ��K	k<���A�G*

logging/current_cost�;��Ԧ+       ��K	<o���A�G*

logging/current_cost�!�;Qio�+       ��K	
����A�G*

logging/current_costf�;D�0+       ��K	�㓔�A�G*

logging/current_cost�\�;@�W+       ��K	���A�G*

logging/current_cost\%�;�8+       ��K	�Q���A�G*

logging/current_cost���;���+       ��K	�}���A�G*

logging/current_cost�~�;<(�	+       ��K	�����A�G*

logging/current_cost�"�;n��I+       ��K	�ߔ��A�G*

logging/current_costG1�;`ׅ�+       ��K	����A�G*

logging/current_cost,T�;���]+       ��K	f>���A�G*

logging/current_cost ��;����+       ��K	�n���A�G*

logging/current_cost�+�;�eC�+       ��K	�����A�G*

logging/current_cost���; ��D+       ��K	�ŕ��A�G*

logging/current_costĻ�;L�0+       ��K	z����A�G*

logging/current_cost�E�;C�ǣ+       ��K	.'���A�G*

logging/current_costd��;�Ԩ+       ��K	DV���A�H*

logging/current_cost�$�;l�]�+       ��K	Ӆ���A�H*

logging/current_costuJ�;l༨+       ��K	����A�H*

logging/current_cost5h�;��|+       ��K	^����A�H*

logging/current_cost�L�;���1+       ��K	(���A�H*

logging/current_cost�R�;U��+       ��K	sC���A�H*

logging/current_cost���;-�<�+       ��K	s���A�H*

logging/current_cost���;s��+       ��K	�����A�H*

logging/current_cost0��;\�+       ��K	v֗��A�H*

logging/current_cost�H�;���V+       ��K	����A�H*

logging/current_cost���;���b+       ��K	d8���A�H*

logging/current_cost���;��Q�+       ��K	^f���A�H*

logging/current_cost ��;�N��+       ��K		����A�H*

logging/current_costu��;��!�+       ��K	����A�H*

logging/current_cost)��;1�w�+       ��K	���A�H*

logging/current_costUP�;e��#+       ��K	����A�H*

logging/current_cost��;�3��+       ��K	yQ���A�H*

logging/current_cost�F�;��+       ��K	݁���A�H*

logging/current_cost$0�;bt=+       ��K	�����A�H*

logging/current_cost���;YƩ�+       ��K	#ޙ��A�H*

logging/current_cost��;eAծ+       ��K	����A�H*

logging/current_costp�;��>+       ��K	J���A�H*

logging/current_cost\h�;$+(i+       ��K	{���A�H*

logging/current_costr��;D��+       ��K	�����A�H*

logging/current_costB��;�W<=+       ��K	�֚��A�H*

logging/current_cost��;��Չ+       ��K	����A�I*

logging/current_costY��;9�S�+       ��K	�2���A�I*

logging/current_cost�&�;Y�F+       ��K	�e���A�I*

logging/current_cost��;^$�+       ��K	+����A�I*

logging/current_costbv�;�AFw+       ��K	�Û��A�I*

logging/current_costD��;���+       ��K	?��A�I*

logging/current_cost���;Q�+       ��K	����A�I*

logging/current_cost�!�;g��]+       ��K	�P���A�I*

logging/current_cost���;3���+       ��K	�����A�I*

logging/current_costr��;�>|g+       ��K	����A�I*

logging/current_cost�N�;�H��+       ��K	�ۜ��A�I*

logging/current_cost@�;n;�+       ��K	�	���A�I*

logging/current_costl��;�dS@+       ��K	7���A�I*

logging/current_costg��;$��@+       ��K	�d���A�I*

logging/current_cost���;�p�[+       ��K	�����A�I*

logging/current_cost��;W�
�+       ��K	�����A�I*

logging/current_cost��;��$	+       ��K	L띔�A�I*

logging/current_cost�$�;(�s(+       ��K	?���A�I*

logging/current_cost9x�;�g�+       ��K	rF���A�I*

logging/current_cost`�;ǥ;E+       ��K	�s���A�I*

logging/current_cost�T�;ys��+       ��K	�����A�I*

logging/current_costkl�;���+       ��K	 ͞��A�I*

logging/current_costn��;R�ܰ+       ��K	�����A�I*

logging/current_cost��;im�+       ��K	�(���A�I*

logging/current_cost���;��ۏ+       ��K	�U���A�I*

logging/current_cost�=�;�w0+       ��K	d����A�I*

logging/current_cost�@�;l��+       ��K	֮���A�J*

logging/current_cost�_�;N���+       ��K	Yܟ��A�J*

logging/current_cost�&�;H_b+       ��K	
���A�J*

logging/current_cost�-�;�X�+       ��K	�?���A�J*

logging/current_cost'2�;��+       ��K	t���A�J*

logging/current_cost��;�x21+       ��K	y����A�J*

logging/current_cost���;�"&h+       ��K	HѠ��A�J*

logging/current_cost�M�;'�G�+       ��K	y����A�J*

logging/current_cost���;��+       ��K	�,���A�J*

logging/current_cost�}�;Lx�+       ��K	Z���A�J*

logging/current_cost���;��mc+       ��K	>����A�J*

logging/current_costU��;��+       ��K	����A�J*

logging/current_cost���;�1��+       ��K	dᡔ�A�J*

logging/current_cost+0�;�)%+       ��K	����A�J*

logging/current_cost�u�;H�`�+       ��K	r<���A�J*

logging/current_cost���;�g��+       ��K	`j���A�J*

logging/current_cost���;L ��+       ��K	Ǚ���A�J*

logging/current_cost%��;Q�+       ��K	Ǣ��A�J*

logging/current_cost.��;\�4�+       ��K	f����A�J*

logging/current_cost��;��|+       ��K	"���A�J*

logging/current_cost�M�;�gd+       ��K	R���A�J*

logging/current_cost�V�;�G ?+       ��K	%����A�J*

logging/current_cost���;zg�\+       ��K	�����A�J*

logging/current_cost���;��,+       ��K	�ޣ��A�J*

logging/current_cost���;S�<+       ��K	����A�J*

logging/current_costW��;���+       ��K	�<���A�K*

logging/current_cost+�;��DN+       ��K	k���A�K*

logging/current_cost;��;����+       ��K	�����A�K*

logging/current_costҸ�;	�+       ��K	qͤ��A�K*

logging/current_cost�V�;�G?�+       ��K	k���A�K*

logging/current_cost�;+��`+       ��K	�7���A�K*

logging/current_coste��;��x+       ��K	�l���A�K*

logging/current_cost|8�;ņ��+       ��K	�����A�K*

logging/current_cost�3�;GO��+       ��K	�˥��A�K*

logging/current_costI	�;6+       ��K	�����A�K*

logging/current_cost ��;�`�+       ��K	k5���A�K*

logging/current_costY�;W05[+       ��K	Tk���A�K*

logging/current_cost���;s��+       ��K	����A�K*

logging/current_costEI�;���i+       ��K	צ��A�K*

logging/current_cost(�;1�
+       ��K	����A�K*

logging/current_cost��;}{/�+       ��K	�E���A�K*

logging/current_cost�}�;$�g+       ��K	�y���A�K*

logging/current_cost�C�;��\}+       ��K	�����A�K*

logging/current_cost�(�;I<�+       ��K	Tߧ��A�K*

logging/current_cost<�;1z�+       ��K	 ���A�K*

logging/current_cost���;Z,1�+       ��K	$C���A�K*

logging/current_cost���;�UR+       ��K	/s���A�K*

logging/current_cost�N�;8�Gd+       ��K	C����A�K*

logging/current_cost�&�;����+       ��K	Hר��A�K*

logging/current_cost���;�i�+       ��K	W���A�K*

logging/current_cost���;�&'h+       ��K	A���A�K*

logging/current_cost|��;!e�<+       ��K	�r���A�L*

logging/current_costN1�;�ô*+       ��K	飩��A�L*

logging/current_cost�@�;�:&%+       ��K	�ԩ��A�L*

logging/current_costU��;X!M�+       ��K	����A�L*

logging/current_cost�\�;�Û�+       ��K	6���A�L*

logging/current_cost��;Y�7L+       ��K	tk���A�L*

logging/current_cost��;�D~+       ��K	]����A�L*

logging/current_costG\�;���+       ��K	1Ϊ��A�L*

logging/current_cost��;lF�+       ��K	� ���A�L*

logging/current_costU��;��-�+       ��K		;���A�L*

logging/current_cost	��;�H�+       ��K	Si���A�L*

logging/current_cost�A�;p�JX+       ��K	홫��A�L*

logging/current_cost�f�;)>Ǚ+       ��K	�Ϋ��A�L*

logging/current_costU�;IOf+       ��K	� ���A�L*

logging/current_cost�;5J�+       ��K	�/���A�L*

logging/current_costK�;�ƦS+       ��K	�a���A�L*

logging/current_cost@��;���+       ��K	E����A�L*

logging/current_cost:�;%e+       ��K	Bɬ��A�L*

logging/current_cost�"�;� ��+       ��K	����A�L*

logging/current_costR��;�|�a+       ��K	�:���A�L*

logging/current_cost���;p�v�+       ��K	�y���A�L*

logging/current_cost){�;�e6�+       ��K	����A�L*

logging/current_cost���;D�?*+       ��K	�䭔�A�L*

logging/current_costui�;֟=�+       ��K	���A�L*

logging/current_cost���;�RIo+       ��K	�F���A�L*

logging/current_cost�Z�;��u+       ��K	�v���A�L*

logging/current_costK��;c�+       ��K	�����A�M*

logging/current_cost���;<'�+       ��K	�����A�M*

logging/current_costU/�;�s�+       ��K	`-���A�M*

logging/current_cost���;��j�+       ��K	qs���A�M*

logging/current_cost��;v���+       ��K	�����A�M*

logging/current_cost ��;Z��+       ��K	�ݯ��A�M*

logging/current_cost˚�;����+       ��K	����A�M*

logging/current_cost�;\���+       ��K	C���A�M*

logging/current_cost W�;��/5+       ��K	�t���A�M*

logging/current_costNd�;��+       ��K	����A�M*

logging/current_cost4�;�{B+       ��K	;ٰ��A�M*

logging/current_costb�;B�+       ��K	����A�M*

logging/current_cost��;���+       ��K	�:���A�M*

logging/current_cost�9�;@�!�+       ��K	�m���A�M*

logging/current_cost�d�;�ײ+       ��K	#����A�M*

logging/current_cost6�;қ�+       ��K	8߱��A�M*

logging/current_costy3�;�C�1+       ��K	����A�M*

logging/current_costG��;��+       ��K	�F���A�M*

logging/current_cost0e�;�IJ+       ��K	�w���A�M*

logging/current_cost��;���+       ��K	�����A�M*

logging/current_cost���;~z��+       ��K	2ݲ��A�M*

logging/current_costK�;H!�%+       ��K	f���A�M*

logging/current_costU��;$�%+       ��K	rU���A�M*

logging/current_cost$��;���+       ��K	Ȅ���A�M*

logging/current_cost���;���+       ��K	c����A�M*

logging/current_cost���;`7V+       ��K	i賔�A�N*

logging/current_cost�$�;���-+       ��K	+���A�N*

logging/current_cost���;�{
j+       ��K	O���A�N*

logging/current_cost<�;BZ��