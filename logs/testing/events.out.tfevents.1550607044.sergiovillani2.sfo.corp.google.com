       �K"	   ��Abrain.Event:2х;M�      ��	D	��A"��
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
9layer_1/weights1/Initializer/random_uniform/RandomUniformRandomUniform1layer_1/weights1/Initializer/random_uniform/shape*

seed *
T0*#
_class
loc:@layer_1/weights1*
seed2 *
dtype0*
_output_shapes

:
�
/layer_1/weights1/Initializer/random_uniform/subSub/layer_1/weights1/Initializer/random_uniform/max/layer_1/weights1/Initializer/random_uniform/min*
T0*#
_class
loc:@layer_1/weights1*
_output_shapes
: 
�
/layer_1/weights1/Initializer/random_uniform/mulMul9layer_1/weights1/Initializer/random_uniform/RandomUniform/layer_1/weights1/Initializer/random_uniform/sub*
_output_shapes

:*
T0*#
_class
loc:@layer_1/weights1
�
+layer_1/weights1/Initializer/random_uniformAdd/layer_1/weights1/Initializer/random_uniform/mul/layer_1/weights1/Initializer/random_uniform/min*
T0*#
_class
loc:@layer_1/weights1*
_output_shapes

:
�
layer_1/weights1
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
layer_1/weights1/AssignAssignlayer_1/weights1+layer_1/weights1/Initializer/random_uniform*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@layer_1/weights1
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
layer_1/biases1/AssignAssignlayer_1/biases1!layer_1/biases1/Initializer/zeros*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
:*
use_locking(
z
layer_1/biases1/readIdentitylayer_1/biases1*
T0*"
_class
loc:@layer_1/biases1*
_output_shapes
:
�
layer_1/MatMulMatMulinput/Placeholderlayer_1/weights1/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
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
/layer_2/weights2/Initializer/random_uniform/mulMul9layer_2/weights2/Initializer/random_uniform/RandomUniform/layer_2/weights2/Initializer/random_uniform/sub*
_output_shapes

:*
T0*#
_class
loc:@layer_2/weights2
�
+layer_2/weights2/Initializer/random_uniformAdd/layer_2/weights2/Initializer/random_uniform/mul/layer_2/weights2/Initializer/random_uniform/min*
_output_shapes

:*
T0*#
_class
loc:@layer_2/weights2
�
layer_2/weights2
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
layer_2/weights2/AssignAssignlayer_2/weights2+layer_2/weights2/Initializer/random_uniform*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@layer_2/weights2
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
layer_2/biases2/AssignAssignlayer_2/biases2!layer_2/biases2/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@layer_2/biases2*
validate_shape(*
_output_shapes
:
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
/layer_3/weights3/Initializer/random_uniform/minConst*#
_class
loc:@layer_3/weights3*
valueB
 *׳]�*
dtype0*
_output_shapes
: 
�
/layer_3/weights3/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *#
_class
loc:@layer_3/weights3*
valueB
 *׳]?
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
/layer_3/weights3/Initializer/random_uniform/subSub/layer_3/weights3/Initializer/random_uniform/max/layer_3/weights3/Initializer/random_uniform/min*
T0*#
_class
loc:@layer_3/weights3*
_output_shapes
: 
�
/layer_3/weights3/Initializer/random_uniform/mulMul9layer_3/weights3/Initializer/random_uniform/RandomUniform/layer_3/weights3/Initializer/random_uniform/sub*
_output_shapes

:*
T0*#
_class
loc:@layer_3/weights3
�
+layer_3/weights3/Initializer/random_uniformAdd/layer_3/weights3/Initializer/random_uniform/mul/layer_3/weights3/Initializer/random_uniform/min*
T0*#
_class
loc:@layer_3/weights3*
_output_shapes

:
�
layer_3/weights3
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@layer_3/weights3
�
layer_3/weights3/AssignAssignlayer_3/weights3+layer_3/weights3/Initializer/random_uniform*
T0*#
_class
loc:@layer_3/weights3*
validate_shape(*
_output_shapes

:*
use_locking(
�
layer_3/weights3/readIdentitylayer_3/weights3*
_output_shapes

:*
T0*#
_class
loc:@layer_3/weights3
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
layer_3/ReluRelulayer_3/add*'
_output_shapes
:���������*
T0
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
*output/weights4/Initializer/random_uniformAdd.output/weights4/Initializer/random_uniform/mul.output/weights4/Initializer/random_uniform/min*
_output_shapes

:*
T0*"
_class
loc:@output/weights4
�
output/weights4
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
VariableV2*
shared_name *!
_class
loc:@output/biases4*
	container *
shape:*
dtype0*
_output_shapes
:
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
valueB"       *
dtype0*
_output_shapes
:
s
	cost/MeanMeancost/SquaredDifference
cost/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
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
train/gradients/FillFilltrain/gradients/Shapetrain/gradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
}
,train/gradients/cost/Mean_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
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
&train/gradients/cost/Mean_grad/Shape_1Shapecost/SquaredDifference*
_output_shapes
:*
T0*
out_type0
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
(train/gradients/cost/Mean_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :
�
&train/gradients/cost/Mean_grad/MaximumMaximum%train/gradients/cost/Mean_grad/Prod_1(train/gradients/cost/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
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
Atrain/gradients/cost/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgs1train/gradients/cost/SquaredDifference_grad/Shape3train/gradients/cost/SquaredDifference_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
2train/gradients/cost/SquaredDifference_grad/scalarConst'^train/gradients/cost/Mean_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
�
/train/gradients/cost/SquaredDifference_grad/mulMul2train/gradients/cost/SquaredDifference_grad/scalar&train/gradients/cost/Mean_grad/truediv*'
_output_shapes
:���������*
T0
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
3train/gradients/cost/SquaredDifference_grad/ReshapeReshape/train/gradients/cost/SquaredDifference_grad/Sum1train/gradients/cost/SquaredDifference_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
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
/train/gradients/cost/SquaredDifference_grad/NegNeg5train/gradients/cost/SquaredDifference_grad/Reshape_1*'
_output_shapes
:���������*
T0
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
:train/gradients/output/add_grad/tuple/control_dependency_1Identity)train/gradients/output/add_grad/Reshape_11^train/gradients/output/add_grad/tuple/group_deps*
_output_shapes
:*
T0*<
_class2
0.loc:@train/gradients/output/add_grad/Reshape_1
�
)train/gradients/output/MatMul_grad/MatMulMatMul8train/gradients/output/add_grad/tuple/control_dependencyoutput/weights4/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b(*
T0
�
+train/gradients/output/MatMul_grad/MatMul_1MatMullayer_3/Relu8train/gradients/output/add_grad/tuple/control_dependency*
T0*
_output_shapes

:*
transpose_a(*
transpose_b( 
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
(train/gradients/layer_3/add_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
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
*train/gradients/layer_3/MatMul_grad/MatMulMatMul9train/gradients/layer_3/add_grad/tuple/control_dependencylayer_3/weights3/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b(*
T0
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
$train/gradients/layer_2/add_grad/SumSum*train/gradients/layer_2/Relu_grad/ReluGrad6train/gradients/layer_2/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
(train/gradients/layer_2/add_grad/ReshapeReshape$train/gradients/layer_2/add_grad/Sum&train/gradients/layer_2/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
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
9train/gradients/layer_2/add_grad/tuple/control_dependencyIdentity(train/gradients/layer_2/add_grad/Reshape2^train/gradients/layer_2/add_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*;
_class1
/-loc:@train/gradients/layer_2/add_grad/Reshape
�
;train/gradients/layer_2/add_grad/tuple/control_dependency_1Identity*train/gradients/layer_2/add_grad/Reshape_12^train/gradients/layer_2/add_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_2/add_grad/Reshape_1*
_output_shapes
:
�
*train/gradients/layer_2/MatMul_grad/MatMulMatMul9train/gradients/layer_2/add_grad/tuple/control_dependencylayer_2/weights2/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b(*
T0
�
,train/gradients/layer_2/MatMul_grad/MatMul_1MatMullayer_1/Relu9train/gradients/layer_2/add_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:*
transpose_a(
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
*train/gradients/layer_1/Relu_grad/ReluGradReluGrad<train/gradients/layer_2/MatMul_grad/tuple/control_dependencylayer_1/Relu*'
_output_shapes
:���������*
T0
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
6train/gradients/layer_1/add_grad/BroadcastGradientArgsBroadcastGradientArgs&train/gradients/layer_1/add_grad/Shape(train/gradients/layer_1/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
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
*train/gradients/layer_1/add_grad/Reshape_1Reshape&train/gradients/layer_1/add_grad/Sum_1(train/gradients/layer_1/add_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
�
1train/gradients/layer_1/add_grad/tuple/group_depsNoOp)^train/gradients/layer_1/add_grad/Reshape+^train/gradients/layer_1/add_grad/Reshape_1
�
9train/gradients/layer_1/add_grad/tuple/control_dependencyIdentity(train/gradients/layer_1/add_grad/Reshape2^train/gradients/layer_1/add_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*;
_class1
/-loc:@train/gradients/layer_1/add_grad/Reshape
�
;train/gradients/layer_1/add_grad/tuple/control_dependency_1Identity*train/gradients/layer_1/add_grad/Reshape_12^train/gradients/layer_1/add_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_1/add_grad/Reshape_1*
_output_shapes
:
�
*train/gradients/layer_1/MatMul_grad/MatMulMatMul9train/gradients/layer_1/add_grad/tuple/control_dependencylayer_1/weights1/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b(*
T0
�
,train/gradients/layer_1/MatMul_grad/MatMul_1MatMulinput/Placeholder9train/gradients/layer_1/add_grad/tuple/control_dependency*
_output_shapes

:*
transpose_a(*
transpose_b( *
T0
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
VariableV2*
shared_name *"
_class
loc:@layer_1/biases1*
	container *
shape: *
dtype0*
_output_shapes
: 
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
$train/layer_1/weights1/Adam_1/AssignAssigntrain/layer_1/weights1/Adam_1/train/layer_1/weights1/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@layer_1/weights1
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
VariableV2*
shared_name *"
_class
loc:@layer_1/biases1*
	container *
shape:*
dtype0*
_output_shapes
:
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
.train/layer_1/biases1/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:*"
_class
loc:@layer_1/biases1*
valueB*    
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
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@layer_2/weights2
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
!train/layer_2/biases2/Adam/AssignAssigntrain/layer_2/biases2/Adam,train/layer_2/biases2/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@layer_2/biases2
�
train/layer_2/biases2/Adam/readIdentitytrain/layer_2/biases2/Adam*
T0*"
_class
loc:@layer_2/biases2*
_output_shapes
:
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
VariableV2*
shared_name *#
_class
loc:@layer_3/weights3*
	container *
shape
:*
dtype0*
_output_shapes

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
!train/layer_3/biases3/Adam/AssignAssigntrain/layer_3/biases3/Adam,train/layer_3/biases3/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@layer_3/biases3
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
#train/layer_3/biases3/Adam_1/AssignAssigntrain/layer_3/biases3/Adam_1.train/layer_3/biases3/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@layer_3/biases3
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
!train/output/weights4/Adam/AssignAssigntrain/output/weights4/Adam,train/output/weights4/Adam/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@output/weights4*
validate_shape(*
_output_shapes

:
�
train/output/weights4/Adam/readIdentitytrain/output/weights4/Adam*
_output_shapes

:*
T0*"
_class
loc:@output/weights4
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
#train/output/weights4/Adam_1/AssignAssigntrain/output/weights4/Adam_1.train/output/weights4/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*"
_class
loc:@output/weights4
�
!train/output/weights4/Adam_1/readIdentitytrain/output/weights4/Adam_1*
_output_shapes

:*
T0*"
_class
loc:@output/weights4
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
train/Adam/beta2Const*
valueB
 *w�?*
dtype0*
_output_shapes
: 
W
train/Adam/epsilonConst*
dtype0*
_output_shapes
: *
valueB
 *w�+2
�
,train/Adam/update_layer_1/weights1/ApplyAdam	ApplyAdamlayer_1/weights1train/layer_1/weights1/Adamtrain/layer_1/weights1/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon>train/gradients/layer_1/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*#
_class
loc:@layer_1/weights1*
use_nesterov( *
_output_shapes

:
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
*train/Adam/update_output/biases4/ApplyAdam	ApplyAdamoutput/biases4train/output/biases4/Adamtrain/output/biases4/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon:train/gradients/output/add_grad/tuple/control_dependency_1*
use_locking( *
T0*!
_class
loc:@output/biases4*
use_nesterov( *
_output_shapes
:
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
train/Adam/mul_1Multrain/beta2_power/readtrain/Adam/beta2,^train/Adam/update_layer_1/biases1/ApplyAdam-^train/Adam/update_layer_1/weights1/ApplyAdam,^train/Adam/update_layer_2/biases2/ApplyAdam-^train/Adam/update_layer_2/weights2/ApplyAdam,^train/Adam/update_layer_3/biases3/ApplyAdam-^train/Adam/update_layer_3/weights3/ApplyAdam+^train/Adam/update_output/biases4/ApplyAdam,^train/Adam/update_output/weights4/ApplyAdam*
_output_shapes
: *
T0*"
_class
loc:@layer_1/biases1
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
save/SaveV2/tensor_namesConst*
dtype0*
_output_shapes
:*�
value�B�Blayer_1/biases1Blayer_1/weights1Blayer_2/biases2Blayer_2/weights2Blayer_3/biases3Blayer_3/weights3Boutput/biases4Boutput/weights4Btrain/beta1_powerBtrain/beta2_powerBtrain/layer_1/biases1/AdamBtrain/layer_1/biases1/Adam_1Btrain/layer_1/weights1/AdamBtrain/layer_1/weights1/Adam_1Btrain/layer_2/biases2/AdamBtrain/layer_2/biases2/Adam_1Btrain/layer_2/weights2/AdamBtrain/layer_2/weights2/Adam_1Btrain/layer_3/biases3/AdamBtrain/layer_3/biases3/Adam_1Btrain/layer_3/weights3/AdamBtrain/layer_3/weights3/Adam_1Btrain/output/biases4/AdamBtrain/output/biases4/Adam_1Btrain/output/weights4/AdamBtrain/output/weights4/Adam_1
�
save/SaveV2/shape_and_slicesConst*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
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
save/RestoreV2/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:*�
value�B�Blayer_1/biases1Blayer_1/weights1Blayer_2/biases2Blayer_2/weights2Blayer_3/biases3Blayer_3/weights3Boutput/biases4Boutput/weights4Btrain/beta1_powerBtrain/beta2_powerBtrain/layer_1/biases1/AdamBtrain/layer_1/biases1/Adam_1Btrain/layer_1/weights1/AdamBtrain/layer_1/weights1/Adam_1Btrain/layer_2/biases2/AdamBtrain/layer_2/biases2/Adam_1Btrain/layer_2/weights2/AdamBtrain/layer_2/weights2/Adam_1Btrain/layer_3/biases3/AdamBtrain/layer_3/biases3/Adam_1Btrain/layer_3/weights3/AdamBtrain/layer_3/weights3/Adam_1Btrain/output/biases4/AdamBtrain/output/biases4/Adam_1Btrain/output/weights4/AdamBtrain/output/weights4/Adam_1
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
save/AssignAssignlayer_1/biases1save/RestoreV2*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@layer_1/biases1
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
save/Assign_3Assignlayer_2/weights2save/RestoreV2:3*
use_locking(*
T0*#
_class
loc:@layer_2/weights2*
validate_shape(*
_output_shapes

:
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
save/Assign_6Assignoutput/biases4save/RestoreV2:6*
T0*!
_class
loc:@output/biases4*
validate_shape(*
_output_shapes
:*
use_locking(
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
save/Assign_15Assigntrain/layer_2/biases2/Adam_1save/RestoreV2:15*
use_locking(*
T0*"
_class
loc:@layer_2/biases2*
validate_shape(*
_output_shapes
:
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
save/Assign_17Assigntrain/layer_2/weights2/Adam_1save/RestoreV2:17*
use_locking(*
T0*#
_class
loc:@layer_2/weights2*
validate_shape(*
_output_shapes

:
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
save/Assign_21Assigntrain/layer_3/weights3/Adam_1save/RestoreV2:21*
T0*#
_class
loc:@layer_3/weights3*
validate_shape(*
_output_shapes

:*
use_locking(
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
save/Assign_23Assigntrain/output/biases4/Adam_1save/RestoreV2:23*
T0*!
_class
loc:@output/biases4*
validate_shape(*
_output_shapes
:*
use_locking(
�
save/Assign_24Assigntrain/output/weights4/Adamsave/RestoreV2:24*
T0*"
_class
loc:@output/weights4*
validate_shape(*
_output_shapes

:*
use_locking(
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
initNoOp^layer_1/biases1/Assign^layer_1/weights1/Assign^layer_2/biases2/Assign^layer_2/weights2/Assign^layer_3/biases3/Assign^layer_3/weights3/Assign^output/biases4/Assign^output/weights4/Assign^train/beta1_power/Assign^train/beta2_power/Assign"^train/layer_1/biases1/Adam/Assign$^train/layer_1/biases1/Adam_1/Assign#^train/layer_1/weights1/Adam/Assign%^train/layer_1/weights1/Adam_1/Assign"^train/layer_2/biases2/Adam/Assign$^train/layer_2/biases2/Adam_1/Assign#^train/layer_2/weights2/Adam/Assign%^train/layer_2/weights2/Adam_1/Assign"^train/layer_3/biases3/Adam/Assign$^train/layer_3/biases3/Adam_1/Assign#^train/layer_3/weights3/Adam/Assign%^train/layer_3/weights3/Adam_1/Assign!^train/output/biases4/Adam/Assign#^train/output/biases4/Adam_1/Assign"^train/output/weights4/Adam/Assign$^train/output/weights4/Adam_1/Assign"j�E�     ��d]	
��AJ܉
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
shape:���������*
dtype0*'
_output_shapes
:���������
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
layer_1/weights1/AssignAssignlayer_1/weights1+layer_1/weights1/Initializer/random_uniform*
T0*#
_class
loc:@layer_1/weights1*
validate_shape(*
_output_shapes

:*
use_locking(
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
layer_1/addAddlayer_1/MatMullayer_1/biases1/read*
T0*'
_output_shapes
:���������
S
layer_1/ReluRelulayer_1/add*'
_output_shapes
:���������*
T0
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
+layer_2/weights2/Initializer/random_uniformAdd/layer_2/weights2/Initializer/random_uniform/mul/layer_2/weights2/Initializer/random_uniform/min*
_output_shapes

:*
T0*#
_class
loc:@layer_2/weights2
�
layer_2/weights2
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
!layer_2/biases2/Initializer/zerosConst*
dtype0*
_output_shapes
:*"
_class
loc:@layer_2/biases2*
valueB*    
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
layer_2/biases2/AssignAssignlayer_2/biases2!layer_2/biases2/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@layer_2/biases2
z
layer_2/biases2/readIdentitylayer_2/biases2*
T0*"
_class
loc:@layer_2/biases2*
_output_shapes
:
�
layer_2/MatMulMatMullayer_1/Relulayer_2/weights2/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
j
layer_2/addAddlayer_2/MatMullayer_2/biases2/read*
T0*'
_output_shapes
:���������
S
layer_2/ReluRelulayer_2/add*'
_output_shapes
:���������*
T0
�
1layer_3/weights3/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*#
_class
loc:@layer_3/weights3*
valueB"      
�
/layer_3/weights3/Initializer/random_uniform/minConst*#
_class
loc:@layer_3/weights3*
valueB
 *׳]�*
dtype0*
_output_shapes
: 
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
seed2 *
dtype0*
_output_shapes

:*

seed *
T0*#
_class
loc:@layer_3/weights3
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
layer_3/weights3/AssignAssignlayer_3/weights3+layer_3/weights3/Initializer/random_uniform*
T0*#
_class
loc:@layer_3/weights3*
validate_shape(*
_output_shapes

:*
use_locking(
�
layer_3/weights3/readIdentitylayer_3/weights3*
T0*#
_class
loc:@layer_3/weights3*
_output_shapes

:
�
!layer_3/biases3/Initializer/zerosConst*
dtype0*
_output_shapes
:*"
_class
loc:@layer_3/biases3*
valueB*    
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
layer_3/biases3/readIdentitylayer_3/biases3*
T0*"
_class
loc:@layer_3/biases3*
_output_shapes
:
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
.output/weights4/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *"
_class
loc:@output/weights4*
valueB
 *qĜ?
�
8output/weights4/Initializer/random_uniform/RandomUniformRandomUniform0output/weights4/Initializer/random_uniform/shape*
T0*"
_class
loc:@output/weights4*
seed2 *
dtype0*
_output_shapes

:*

seed 
�
.output/weights4/Initializer/random_uniform/subSub.output/weights4/Initializer/random_uniform/max.output/weights4/Initializer/random_uniform/min*
T0*"
_class
loc:@output/weights4*
_output_shapes
: 
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
shape:*
dtype0*
_output_shapes
:*
shared_name *!
_class
loc:@output/biases4*
	container 
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
valueB"       *
dtype0*
_output_shapes
:
s
	cost/MeanMeancost/SquaredDifference
cost/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
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
train/gradients/FillFilltrain/gradients/Shapetrain/gradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
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
&train/gradients/cost/Mean_grad/Shape_1Shapecost/SquaredDifference*
_output_shapes
:*
T0*
out_type0
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
#train/gradients/cost/Mean_grad/ProdProd&train/gradients/cost/Mean_grad/Shape_1$train/gradients/cost/Mean_grad/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
p
&train/gradients/cost/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
%train/gradients/cost/Mean_grad/Prod_1Prod&train/gradients/cost/Mean_grad/Shape_2&train/gradients/cost/Mean_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
j
(train/gradients/cost/Mean_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :
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
&train/gradients/cost/Mean_grad/truedivRealDiv#train/gradients/cost/Mean_grad/Tile#train/gradients/cost/Mean_grad/Cast*'
_output_shapes
:���������*
T0
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
Atrain/gradients/cost/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgs1train/gradients/cost/SquaredDifference_grad/Shape3train/gradients/cost/SquaredDifference_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
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
/train/gradients/cost/SquaredDifference_grad/SumSum1train/gradients/cost/SquaredDifference_grad/mul_1Atrain/gradients/cost/SquaredDifference_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
3train/gradients/cost/SquaredDifference_grad/ReshapeReshape/train/gradients/cost/SquaredDifference_grad/Sum1train/gradients/cost/SquaredDifference_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
1train/gradients/cost/SquaredDifference_grad/Sum_1Sum1train/gradients/cost/SquaredDifference_grad/mul_1Ctrain/gradients/cost/SquaredDifference_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
5train/gradients/cost/SquaredDifference_grad/Reshape_1Reshape1train/gradients/cost/SquaredDifference_grad/Sum_13train/gradients/cost/SquaredDifference_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
/train/gradients/cost/SquaredDifference_grad/NegNeg5train/gradients/cost/SquaredDifference_grad/Reshape_1*'
_output_shapes
:���������*
T0
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
#train/gradients/output/add_grad/SumSumDtrain/gradients/cost/SquaredDifference_grad/tuple/control_dependency5train/gradients/output/add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
'train/gradients/output/add_grad/ReshapeReshape#train/gradients/output/add_grad/Sum%train/gradients/output/add_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
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
$train/gradients/layer_3/add_grad/SumSum*train/gradients/layer_3/Relu_grad/ReluGrad6train/gradients/layer_3/add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
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
*train/gradients/layer_3/add_grad/Reshape_1Reshape&train/gradients/layer_3/add_grad/Sum_1(train/gradients/layer_3/add_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
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
;train/gradients/layer_3/add_grad/tuple/control_dependency_1Identity*train/gradients/layer_3/add_grad/Reshape_12^train/gradients/layer_3/add_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_3/add_grad/Reshape_1*
_output_shapes
:
�
*train/gradients/layer_3/MatMul_grad/MatMulMatMul9train/gradients/layer_3/add_grad/tuple/control_dependencylayer_3/weights3/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b(*
T0
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
$train/gradients/layer_2/add_grad/SumSum*train/gradients/layer_2/Relu_grad/ReluGrad6train/gradients/layer_2/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
(train/gradients/layer_2/add_grad/ReshapeReshape$train/gradients/layer_2/add_grad/Sum&train/gradients/layer_2/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
&train/gradients/layer_2/add_grad/Sum_1Sum*train/gradients/layer_2/Relu_grad/ReluGrad8train/gradients/layer_2/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
*train/gradients/layer_2/add_grad/Reshape_1Reshape&train/gradients/layer_2/add_grad/Sum_1(train/gradients/layer_2/add_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
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
*train/gradients/layer_2/MatMul_grad/MatMulMatMul9train/gradients/layer_2/add_grad/tuple/control_dependencylayer_2/weights2/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b(*
T0
�
,train/gradients/layer_2/MatMul_grad/MatMul_1MatMullayer_1/Relu9train/gradients/layer_2/add_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:*
transpose_a(
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
&train/gradients/layer_1/add_grad/ShapeShapelayer_1/MatMul*
_output_shapes
:*
T0*
out_type0
r
(train/gradients/layer_1/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
6train/gradients/layer_1/add_grad/BroadcastGradientArgsBroadcastGradientArgs&train/gradients/layer_1/add_grad/Shape(train/gradients/layer_1/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
$train/gradients/layer_1/add_grad/SumSum*train/gradients/layer_1/Relu_grad/ReluGrad6train/gradients/layer_1/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
(train/gradients/layer_1/add_grad/ReshapeReshape$train/gradients/layer_1/add_grad/Sum&train/gradients/layer_1/add_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
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
9train/gradients/layer_1/add_grad/tuple/control_dependencyIdentity(train/gradients/layer_1/add_grad/Reshape2^train/gradients/layer_1/add_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*;
_class1
/-loc:@train/gradients/layer_1/add_grad/Reshape
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
VariableV2*
shared_name *"
_class
loc:@layer_1/biases1*
	container *
shape: *
dtype0*
_output_shapes
: 
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
train/beta1_power/readIdentitytrain/beta1_power*
_output_shapes
: *
T0*"
_class
loc:@layer_1/biases1
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
VariableV2*#
_class
loc:@layer_1/weights1*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
�
"train/layer_1/weights1/Adam/AssignAssigntrain/layer_1/weights1/Adam-train/layer_1/weights1/Adam/Initializer/zeros*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@layer_1/weights1
�
 train/layer_1/weights1/Adam/readIdentitytrain/layer_1/weights1/Adam*
T0*#
_class
loc:@layer_1/weights1*
_output_shapes

:
�
/train/layer_1/weights1/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes

:*#
_class
loc:@layer_1/weights1*
valueB*    
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
$train/layer_1/weights1/Adam_1/AssignAssigntrain/layer_1/weights1/Adam_1/train/layer_1/weights1/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@layer_1/weights1
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
VariableV2*
shared_name *"
_class
loc:@layer_1/biases1*
	container *
shape:*
dtype0*
_output_shapes
:
�
!train/layer_1/biases1/Adam/AssignAssigntrain/layer_1/biases1/Adam,train/layer_1/biases1/Adam/Initializer/zeros*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
:*
use_locking(
�
train/layer_1/biases1/Adam/readIdentitytrain/layer_1/biases1/Adam*
T0*"
_class
loc:@layer_1/biases1*
_output_shapes
:
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
/train/layer_2/weights2/Adam_1/Initializer/zerosConst*#
_class
loc:@layer_2/weights2*
valueB*    *
dtype0*
_output_shapes

:
�
train/layer_2/weights2/Adam_1
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
$train/layer_2/weights2/Adam_1/AssignAssigntrain/layer_2/weights2/Adam_1/train/layer_2/weights2/Adam_1/Initializer/zeros*
T0*#
_class
loc:@layer_2/weights2*
validate_shape(*
_output_shapes

:*
use_locking(
�
"train/layer_2/weights2/Adam_1/readIdentitytrain/layer_2/weights2/Adam_1*
T0*#
_class
loc:@layer_2/weights2*
_output_shapes

:
�
,train/layer_2/biases2/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:*"
_class
loc:@layer_2/biases2*
valueB*    
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
-train/layer_3/weights3/Adam/Initializer/zerosConst*#
_class
loc:@layer_3/weights3*
valueB*    *
dtype0*
_output_shapes

:
�
train/layer_3/weights3/Adam
VariableV2*
shared_name *#
_class
loc:@layer_3/weights3*
	container *
shape
:*
dtype0*
_output_shapes

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
VariableV2*
shared_name *#
_class
loc:@layer_3/weights3*
	container *
shape
:*
dtype0*
_output_shapes

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
,train/layer_3/biases3/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:*"
_class
loc:@layer_3/biases3*
valueB*    
�
train/layer_3/biases3/Adam
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@layer_3/biases3
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
train/layer_3/biases3/Adam/readIdentitytrain/layer_3/biases3/Adam*
T0*"
_class
loc:@layer_3/biases3*
_output_shapes
:
�
.train/layer_3/biases3/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:*"
_class
loc:@layer_3/biases3*
valueB*    
�
train/layer_3/biases3/Adam_1
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
#train/layer_3/biases3/Adam_1/AssignAssigntrain/layer_3/biases3/Adam_1.train/layer_3/biases3/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@layer_3/biases3
�
!train/layer_3/biases3/Adam_1/readIdentitytrain/layer_3/biases3/Adam_1*
_output_shapes
:*
T0*"
_class
loc:@layer_3/biases3
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
"train/output/biases4/Adam_1/AssignAssigntrain/output/biases4/Adam_1-train/output/biases4/Adam_1/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@output/biases4*
validate_shape(*
_output_shapes
:
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
+train/Adam/update_layer_1/biases1/ApplyAdam	ApplyAdamlayer_1/biases1train/layer_1/biases1/Adamtrain/layer_1/biases1/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon;train/gradients/layer_1/add_grad/tuple/control_dependency_1*
use_locking( *
T0*"
_class
loc:@layer_1/biases1*
use_nesterov( *
_output_shapes
:
�
,train/Adam/update_layer_2/weights2/ApplyAdam	ApplyAdamlayer_2/weights2train/layer_2/weights2/Adamtrain/layer_2/weights2/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon>train/gradients/layer_2/MatMul_grad/tuple/control_dependency_1*
T0*#
_class
loc:@layer_2/weights2*
use_nesterov( *
_output_shapes

:*
use_locking( 
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
,train/Adam/update_layer_3/weights3/ApplyAdam	ApplyAdamlayer_3/weights3train/layer_3/weights3/Adamtrain/layer_3/weights3/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon>train/gradients/layer_3/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*#
_class
loc:@layer_3/weights3*
use_nesterov( *
_output_shapes

:
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
+train/Adam/update_output/weights4/ApplyAdam	ApplyAdamoutput/weights4train/output/weights4/Adamtrain/output/weights4/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon=train/gradients/output/MatMul_grad/tuple/control_dependency_1*
T0*"
_class
loc:@output/weights4*
use_nesterov( *
_output_shapes

:*
use_locking( 
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
save/SaveV2/shape_and_slicesConst*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
�
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_sliceslayer_1/biases1layer_1/weights1layer_2/biases2layer_2/weights2layer_3/biases3layer_3/weights3output/biases4output/weights4train/beta1_powertrain/beta2_powertrain/layer_1/biases1/Adamtrain/layer_1/biases1/Adam_1train/layer_1/weights1/Adamtrain/layer_1/weights1/Adam_1train/layer_2/biases2/Adamtrain/layer_2/biases2/Adam_1train/layer_2/weights2/Adamtrain/layer_2/weights2/Adam_1train/layer_3/biases3/Adamtrain/layer_3/biases3/Adam_1train/layer_3/weights3/Adamtrain/layer_3/weights3/Adam_1train/output/biases4/Adamtrain/output/biases4/Adam_1train/output/weights4/Adamtrain/output/weights4/Adam_1*(
dtypes
2
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
_output_shapes
: *
T0*
_class
loc:@save/Const
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
save/Assign_5Assignlayer_3/weights3save/RestoreV2:5*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@layer_3/weights3
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
save/Assign_7Assignoutput/weights4save/RestoreV2:7*
use_locking(*
T0*"
_class
loc:@output/weights4*
validate_shape(*
_output_shapes

:
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
save/Assign_11Assigntrain/layer_1/biases1/Adam_1save/RestoreV2:11*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
:*
use_locking(
�
save/Assign_12Assigntrain/layer_1/weights1/Adamsave/RestoreV2:12*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@layer_1/weights1
�
save/Assign_13Assigntrain/layer_1/weights1/Adam_1save/RestoreV2:13*
use_locking(*
T0*#
_class
loc:@layer_1/weights1*
validate_shape(*
_output_shapes

:
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
save/Assign_15Assigntrain/layer_2/biases2/Adam_1save/RestoreV2:15*
use_locking(*
T0*"
_class
loc:@layer_2/biases2*
validate_shape(*
_output_shapes
:
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
save/Assign_21Assigntrain/layer_3/weights3/Adam_1save/RestoreV2:21*
T0*#
_class
loc:@layer_3/weights3*
validate_shape(*
_output_shapes

:*
use_locking(
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
save/Assign_23Assigntrain/output/biases4/Adam_1save/RestoreV2:23*
T0*!
_class
loc:@output/biases4*
validate_shape(*
_output_shapes
:*
use_locking(
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
save/Assign_25Assigntrain/output/weights4/Adam_1save/RestoreV2:25*
T0*"
_class
loc:@output/weights4*
validate_shape(*
_output_shapes

:*
use_locking(
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
train/output/biases4/Adam_1:0"train/output/biases4/Adam_1/Assign"train/output/biases4/Adam_1/read:02/train/output/biases4/Adam_1/Initializer/zeros:0j~�i(       �pJ	����A*

logging/current_cost�m0=DsI�*       ����	Z���A*

logging/current_cost��+=��*       ����	u��A
*

logging/current_costT'=w��*       ����	zO��A*

logging/current_cost�U#=|<L�*       ����	����A*

logging/current_cost[�=ϡ*       ����	���A*

logging/current_cost%=��?�*       ����	����A*

logging/current_cost��=�Ѵ�*       ����	F ��A#*

logging/current_cost��=�*       ����	(V��A(*

logging/current_cost^=��s�*       ����	���A-*

logging/current_cost�=�N6m*       ����	����A2*

logging/current_cost�W=�<�*       ����	 ���A7*

logging/current_cost�P=m�*       ����	��A<*

logging/current_cost�
=7��8*       ����	YG��AA*

logging/current_costi�=�,�*       ����	�t��AF*

logging/current_cost�p=B��*       ����	W���AK*

logging/current_cost�*=ei�*       ����	����AP*

logging/current_cost=���I*       ����	7���AU*

logging/current_cost�=r��H*       ����	{.��AZ*

logging/current_cost�/=X���*       ����	]��A_*

logging/current_costn=�~�*       ����	Ê��Ad*

logging/current_cost��=�+�\*       ����	����Ai*

logging/current_cost�/=�ZL/*       ����	G���An*

logging/current_cost� =�I��*       ����	'��As*

logging/current_cost =�1��*       ����	�D��Ax*

logging/current_cost���<�P*       ����	�s��A}*

logging/current_cost���<a��+       ��K	*���A�*

logging/current_costrO�<�|�+       ��K	����A�*

logging/current_costЍ�<ʜr+       ��K	���A�*

logging/current_cost��<׿�a+       ��K	,��A�*

logging/current_cost0��<��^�+       ��K	\��A�*

logging/current_cost�W�<����+       ��K	I���A�*

logging/current_costr��<QH+       ��K	]���A�*

logging/current_cost�4�<���J+       ��K	����A�*

logging/current_cost��<'3 +       ��K	���A�*

logging/current_cost��<��@i+       ��K	&@��A�*

logging/current_costTX�<��Q+       ��K	�l��A�*

logging/current_cost���<+�+       ��K	����A�*

logging/current_cost|��<�{�P+       ��K	���A�*

logging/current_cost�.�<�y�+       ��K	L���A�*

logging/current_costK��<��c+       ��K	�+��A�*

logging/current_cost'	�<�i%�+       ��K	�\��A�*

logging/current_cost04�<��+       ��K	"���A�*

logging/current_cost��<�~�+       ��K	ض��A�*

logging/current_costm,�<��+       ��K	����A�*

logging/current_cost���<�Tw�+       ��K	$��A�*

logging/current_cost	]�<5ҫ�+       ��K	�>��A�*

logging/current_cost
�<�#+       ��K	�l��A�*

logging/current_cost��<����+       ��K	k���A�*

logging/current_cost���<�(o'+       ��K	7���A�*

logging/current_costU��<�1�t+       ��K	����A�*

logging/current_cost���<��U+       ��K	�'��A�*

logging/current_cost�Ʃ<>R��+       ��K	|U��A�*

logging/current_cost֦<j�_+       ��K	����A�*

logging/current_cost2�<�Oq7+       ��K	����A�*

logging/current_costK�<"d�+       ��K	����A�*

logging/current_cost>=�<9̛4+       ��K	���A�*

logging/current_cost���<�Wzs+       ��K	�5��A�*

logging/current_cost��<�_̠+       ��K	b��A�*

logging/current_cost~�<I37+       ��K	B���A�*

logging/current_cost��<1F�+       ��K	L���A�*

logging/current_cost ��<�u �+       ��K	U���A�*

logging/current_cost�O�<>�NU+       ��K	���A�*

logging/current_cost|�<m�W�+       ��K	�C��A�*

logging/current_cost���<����+       ��K		r��A�*

logging/current_cost�͈<L���+       ��K	H���A�*

logging/current_cost���<=sLl+       ��K	����A�*

logging/current_cost	��<��Da+       ��K	����A�*

logging/current_cost�ق<�oc�+       ��K	/(��A�*

logging/current_cost���<�c�+       ��K	[W��A�*

logging/current_costU8~<~��+       ��K	N���A�*

logging/current_cost�z<[��?+       ��K	���A�*

logging/current_cost;sw< ܰ
+       ��K	+���A�*

logging/current_cost`&t<�I+       ��K	���A�*

logging/current_cost��p<�`?+       ��K	�@��A�*

logging/current_cost"�m<�c��+       ��K	�o��A�*

logging/current_cost��j<DB��+       ��K	P���A�*

logging/current_cost�g<ӊ\+       ��K	8���A�*

logging/current_cost��d<]*��+       ��K	����A�*

logging/current_cost#-b<�?�+       ��K	�+��A�*

logging/current_cost�:_<ë+       ��K	�[��A�*

logging/current_cost�\<g� �+       ��K	���A�*

logging/current_cost�Z< �M�+       ��K	���A�*

logging/current_costw5W<ײe+       ��K	����A�*

logging/current_costיT<	��+       ��K	���A�*

logging/current_costrPR<Q�?�+       ��K	�L��A�*

logging/current_costL�O<=[H�+       ��K	�{��A�*

logging/current_costl�M<xK��+       ��K	S���A�*

logging/current_cost��K<fY�+       ��K	{���A�*

logging/current_costcI<���-+       ��K	��A�*

logging/current_cost;+G<+�=�+       ��K	�3��A�*

logging/current_cost��D<U�v9+       ��K	�a��A�*

logging/current_cost"C<��u+       ��K	S���A�*

logging/current_cost�!A<:H�+       ��K	����A�*

logging/current_cost}?<��n�+       ��K	����A�*

logging/current_cost�=<�V�+       ��K	**��A�*

logging/current_cost�G;<� �+       ��K	!Y��A�*

logging/current_costU�9<���Y+       ��K	���A�*

logging/current_cost�8<Ơ +       ��K	 ���A�*

logging/current_cost�m7<;��+       ��K	���A�*

logging/current_costd�5<w"`q+       ��K	^��A�*

logging/current_cost�4<���+       ��K	�A��A�*

logging/current_cost/�3<L�O[+       ��K	ip��A�*

logging/current_cost~f3<.�@�+       ��K	{���A�*

logging/current_cost�G2<�+Xn+       ��K		���A�*

logging/current_costR$1<X�̔+       ��K	����A�*

logging/current_costّ/<ӣk+       ��K	,' ��A�*

logging/current_cost��.<��O+       ��K	�T ��A�*

logging/current_cost��-<�x+       ��K	Q� ��A�*

logging/current_costim,<�k��+       ��K	�� ��A�*

logging/current_cost�*<�/]+       ��K	k� ��A�*

logging/current_costQ)<����+       ��K	�!��A�*

logging/current_cost`�'<z!�+       ��K	�<!��A�*

logging/current_costP�&<���+       ��K	8j!��A�*

logging/current_cost`W%<��p+       ��K	��!��A�*

logging/current_cost@$<rm�+       ��K	��!��A�*

logging/current_cost #<��dN+       ��K	�!��A�*

logging/current_costC"<�4 �+       ��K	n"��A�*

logging/current_cost 5!<Ccp�+       ��K	CM"��A�*

logging/current_cost�f <!m��+       ��K	$z"��A�*

logging/current_cost��<�1 +       ��K	2�"��A�*

logging/current_cost��<��'d+       ��K	 �"��A�*

logging/current_costhC<i�w�+       ��K	p#��A�*

logging/current_cost�$<}r�^+       ��K	V/#��A�*

logging/current_cost�<*���+       ��K	�[#��A�*

logging/current_costS*<�:�V+       ��K	`�#��A�*

logging/current_cost�p<P�#�+       ��K	(�#��A�*

logging/current_costن<� ��+       ��K	��#��A�*

logging/current_cost1�<g ��+       ��K	Y$��A�*

logging/current_cost5�<l�	�+       ��K	�=$��A�*

logging/current_cost��<��̒+       ��K	�j$��A�*

logging/current_cost)<ý�+       ��K	��$��A�*

logging/current_cost�[<1�u+       ��K	#�$��A�*

logging/current_costd�<�+s�+       ��K	��$��A�*

logging/current_cost�D<Y65�+       ��K	4$%��A�*

logging/current_cost<�߷'+       ��K	iQ%��A�*

logging/current_cost<C���+       ��K	�}%��A�*

logging/current_costFT<�8w�+       ��K	g�%��A�*

logging/current_costL"<Q;P�+       ��K	��%��A�*

logging/current_cost��<CM�+       ��K	�&��A�*

logging/current_cost�<�+�v+       ��K	�4&��A�*

logging/current_costN�<�@�#+       ��K	�c&��A�*

logging/current_cost�<SB�B+       ��K	��&��A�*

logging/current_cost��<H+       ��K	~�&��A�*

logging/current_cost"<.���+       ��K	'��A�*

logging/current_cost��<2+       ��K	mN'��A�*

logging/current_cost{X<6��k+       ��K	�z'��A�*

logging/current_cost�<��Z+       ��K	3�'��A�*

logging/current_costB`<�.� +       ��K	��'��A�*

logging/current_cost�r<�֩+       ��K	t	(��A�*

logging/current_cost�<N��B+       ��K	06(��A�*

logging/current_costY�< �E�+       ��K	cf(��A�*

logging/current_costy<���1+       ��K	(��A�*

logging/current_cost��<��+       ��K	��(��A�*

logging/current_cost��<� �+       ��K	��(��A�*

logging/current_cost�u<��+       ��K	�#)��A�*

logging/current_cost@<���+       ��K	�T)��A�*

logging/current_cost�<�kI.+       ��K	S�)��A�*

logging/current_cost��<��u+       ��K	��)��A�*

logging/current_cost��<�Nr+       ��K	��)��A�*

logging/current_costL5<F�#+       ��K	�*��A�*

logging/current_costA<u\�i+       ��K	�<*��A�*

logging/current_cost_<>�AC+       ��K	j*��A�*

logging/current_cost4C<�z?y+       ��K	.�*��A�*

logging/current_cost�M<���+       ��K	*�*��A�*

logging/current_cost6<�~��+       ��K	_�*��A�*

logging/current_cost��<�i�+       ��K	D$+��A�*

logging/current_costq�<+�T+       ��K	�R+��A�*

logging/current_costb<�k�+       ��K	I�+��A�*

logging/current_cost J<�gv�+       ��K	�+��A�*

logging/current_cost�?<�w�+       ��K	��+��A�*

logging/current_cost<��+       ��K	,��A�*

logging/current_cost�<+1�+       ��K	<,��A�*

logging/current_cost�X<hP�|+       ��K	�i,��A�*

logging/current_cost&�<�,E +       ��K	ז,��A�*

logging/current_cost�N<
+��+       ��K	Z�,��A�*

logging/current_costpb<�,+       ��K	��,��A�*

logging/current_cost�d<�7�+       ��K	k!-��A�*

logging/current_cost�U<�g]�+       ��K	*N-��A�*

logging/current_cost*]<�H+       ��K	�z-��A�*

logging/current_cost�%<q�p*+       ��K	�-��A�*

logging/current_cost�9<� =+       ��K	��-��A�*

logging/current_cost�'<�{�+       ��K	Q
.��A�*

logging/current_cost<<|�E3+       ��K	�8.��A�*

logging/current_costy<f�2+       ��K	�e.��A�*

logging/current_cost�t<���+       ��K	ϔ.��A�*

logging/current_costC^<�$�+       ��K	��.��A�*

logging/current_cost�i<2 �+       ��K	��.��A�*

logging/current_cost�i<fJ�T+       ��K	 /��A�*

logging/current_costiV<A*��+       ��K	�I/��A�*

logging/current_costZK<��	+       ��K	
x/��A�*

logging/current_cost�J<86�.+       ��K	a�/��A�*

logging/current_cost�<�3}�+       ��K	��/��A�*

logging/current_cost\�<(��+       ��K	��/��A�*

logging/current_costh<J��L+       ��K	)0��A�*

logging/current_cost�<s��2+       ��K	�U0��A�*

logging/current_costn�<`�+       ��K	��0��A�*

logging/current_cost�<�	�+       ��K	��0��A�*

logging/current_cost�<kң$+       ��K	b�0��A�*

logging/current_cost81<��Δ+       ��K	1��A�*

logging/current_cost%;<�+Y+       ��K	|=1��A�*

logging/current_cost�!<�mo�+       ��K	Hj1��A�*

logging/current_cost�W<j�'J+       ��K	��1��A�*

logging/current_cost~�<ڛ7�+       ��K	x�1��A�*

logging/current_costM�<6���+       ��K	��1��A�*

logging/current_costL�<嶚+       ��K	B 2��A�*

logging/current_cost�<g+       ��K	AL2��A�*

logging/current_costN�<2�O#+       ��K	 z2��A�*

logging/current_costw�<$��+       ��K	ͧ2��A�*

logging/current_cost0�<���+       ��K	t�2��A�*

logging/current_cost<B��A+       ��K	�3��A�*

logging/current_cost�<����+       ��K	u/3��A�*

logging/current_cost�T<Z(*�+       ��K	�[3��A�*

logging/current_costR�<A��T+       ��K	�3��A�*

logging/current_cost��<9�P�+       ��K	��3��A�*

logging/current_costj4<ǔVX+       ��K	F�3��A�*

logging/current_cost"@<W-k�+       ��K	g4��A�*

logging/current_cost��
<��,+       ��K	H4��A�*

logging/current_cost|�
<��#M+       ��K	{w4��A�*

logging/current_cost�|
<Rl��+       ��K	a�4��A�*

logging/current_cost�&
<���+       ��K	��4��A�*

logging/current_cost��	<?fbM+       ��K	5��A�*

logging/current_costT�	<AA�+       ��K	C/5��A�*

logging/current_costn�	<ؽ�+       ��K	�\5��A�*

logging/current_cost1m	<o�1+       ��K	��5��A�*

logging/current_costU&	<X}��+       ��K	I�5��A�*

logging/current_cost�	<#ͧ�+       ��K	��5��A�*

logging/current_cost��<`%$+       ��K	A6��A�*

logging/current_costg�<Ё1�+       ��K	�E6��A�*

logging/current_cost��<�`<+       ��K	�u6��A�*

logging/current_costq<���+       ��K	ݢ6��A�*

logging/current_cost~e<�wP�+       ��K	��6��A�	*

logging/current_cost�F<�T+       ��K	� 7��A�	*

logging/current_costY5<�ot+       ��K	�07��A�	*

logging/current_cost�+<��:+       ��K	�\7��A�	*

logging/current_costu<��v+       ��K	4�7��A�	*

logging/current_cost�<?�t�+       ��K	��7��A�	*

logging/current_cost��<�b+       ��K	��7��A�	*

logging/current_cost�<�;�+       ��K	�8��A�	*

logging/current_cost��<��f�+       ��K	�C8��A�	*

logging/current_costwd<yp��+       ��K	�t8��A�	*

logging/current_costJ`<�x�1+       ��K	|�8��A�	*

logging/current_costE<����+       ��K	O�8��A�	*

logging/current_cost�(<d� +       ��K	K 9��A�	*

logging/current_cost><6��+       ��K	�19��A�	*

logging/current_costM<k���+       ��K	a9��A�	*

logging/current_cost@�<5Ȇ+       ��K	Q�9��A�	*

logging/current_cost��<�vZ/+       ��K	��9��A�	*

logging/current_cost��<.q��+       ��K	��9��A�	*

logging/current_cost�i<��1+       ��K	V:��A�	*

logging/current_cost�r<A��a+       ��K	rI:��A�	*

logging/current_cost�E<O'�+       ��K	w:��A�	*

logging/current_cost41<m�ϓ+       ��K	��:��A�	*

logging/current_cost�<����+       ��K	��:��A�	*

logging/current_cost��<[��+       ��K	  ;��A�	*

logging/current_costO�<MN�o+       ��K	P/;��A�	*

logging/current_costd�<�-y�+       ��K	^;��A�
*

logging/current_cost�<�w �+       ��K	�;��A�
*

logging/current_cost��<{�+       ��K	�;��A�
*

logging/current_costR�<ۘ�+       ��K	|:<��A�
*

logging/current_cost��<��t�+       ��K	��<��A�
*

logging/current_cost��<���Q+       ��K	-�<��A�
*

logging/current_cost$�<�Z=+       ��K	;=��A�
*

logging/current_cost,�<Q�i�+       ��K	�B=��A�
*

logging/current_cost��<q��+       ��K	�}=��A�
*

logging/current_cost��<��u+       ��K	��=��A�
*

logging/current_costE{<��t+       ��K	�=��A�
*

logging/current_costy}<C��+       ��K	�>��A�
*

logging/current_costL�<����+       ��K	�Q>��A�
*

logging/current_cost�<�˶�+       ��K	��>��A�
*

logging/current_cost\z<3ߐ+       ��K	h�>��A�
*

logging/current_cost�y<���+       ��K	��>��A�
*

logging/current_costw�<TFvS+       ��K	C?��A�
*

logging/current_cost�<z�ƺ+       ��K	�G?��A�
*

logging/current_cost��<<l>�+       ��K	@z?��A�
*

logging/current_cost�v<1Y+       ��K	�?��A�
*

logging/current_cost��<<���+       ��K	v�?��A�
*

logging/current_costB�<�C�c+       ��K	�@��A�
*

logging/current_cost>8<[K+       ��K	p<@��A�
*

logging/current_cost�V<VBY�+       ��K	Oj@��A�
*

logging/current_cost]<շCi+       ��K	.�@��A�
*

logging/current_costu�<v��+       ��K	�@��A�
*

logging/current_cost��<��-=+       ��K	��@��A�*

logging/current_costz�<��.�+       ��K	0A��A�*

logging/current_cost�<ӌ�z+       ��K	`_A��A�*

logging/current_cost�<�;n5+       ��K	�A��A�*

logging/current_costt<����+       ��K	��A��A�*

logging/current_cost <�P�+       ��K	{�A��A�*

logging/current_cost��<�K��+       ��K	�B��A�*

logging/current_cost9�<�(�+       ��K	�IB��A�*

logging/current_costܹ<�M�"+       ��K	`�B��A�*

logging/current_costי<���+       ��K	5�B��A�*

logging/current_cost�<���+       ��K	��B��A�*

logging/current_costt�<C�ݤ+       ��K	�C��A�*

logging/current_costM�<��+       ��K	SGC��A�*

logging/current_costM�<G\U+       ��K	ـC��A�*

logging/current_costt]<=�[a+       ��K	@�C��A�*

logging/current_cost�T<�N��+       ��K	��C��A�*

logging/current_costW_<���+       ��K	`D��A�*

logging/current_costU]<)6
d+       ��K	�>D��A�*

logging/current_costPQ<7�]�+       ��K	lD��A�*

logging/current_cost1<�pD�+       ��K	՜D��A�*

logging/current_cost�*<�cB�+       ��K	��D��A�*

logging/current_cost�@<G+       ��K	��D��A�*

logging/current_costD<FN �+       ��K	r.E��A�*

logging/current_cost�7<�`?|+       ��K	v_E��A�*

logging/current_cost�#<���+       ��K	ُE��A�*

logging/current_cost�<є��+       ��K	�E��A�*

logging/current_cost�<ǲ�+       ��K	�E��A�*

logging/current_cost�<�B�+       ��K	�F��A�*

logging/current_coste<��`Q+       ��K	>HF��A�*

logging/current_costR<��R{+       ��K	~vF��A�*

logging/current_cost~�<Aq�+       ��K	*�F��A�*

logging/current_cost�<�
E�+       ��K	��F��A�*

logging/current_cost<<�j�d+       ��K	G��A�*

logging/current_cost�<���
+       ��K	54G��A�*

logging/current_cost�<�;+       ��K	�aG��A�*

logging/current_cost��<Ɩ�l+       ��K	h�G��A�*

logging/current_cost�<]��S+       ��K	��G��A�*

logging/current_costr<�+       ��K	��G��A�*

logging/current_cost%<�'+       ��K	�H��A�*

logging/current_cost3	<�/�%+       ��K	�DH��A�*

logging/current_cost�</"��+       ��K	*sH��A�*

logging/current_cost�<�<�+       ��K	�H��A�*

logging/current_cost<8�P�+       ��K	��H��A�*

logging/current_cost�<a[�+       ��K	!�H��A�*

logging/current_cost�<����+       ��K	-I��A�*

logging/current_cost��<��w+       ��K	�]I��A�*

logging/current_cost��<-��<+       ��K	i�I��A�*

logging/current_cost�<"��[+       ��K	��I��A�*

logging/current_cost<l�R+       ��K	��I��A�*

logging/current_costI <-Z��+       ��K	�J��A�*

logging/current_cost�<�ݺ�+       ��K	�CJ��A�*

logging/current_cost3
<Q��+       ��K	;rJ��A�*

logging/current_cost[<�`+       ��K	V�J��A�*

logging/current_cost�#<��+       ��K	��J��A�*

logging/current_cost$<6U�/+       ��K	,�J��A�*

logging/current_cost<�ڽ�+       ��K	�+K��A�*

logging/current_cost�<�B5$+       ��K	�XK��A�*

logging/current_cost�%<�M�v+       ��K	~�K��A�*

logging/current_cost�-<���+       ��K	ѵK��A�*

logging/current_cost0<1��+       ��K	��K��A�*

logging/current_cost�/<�I ++       ��K	CL��A�*

logging/current_cost�*<e��K+       ��K	�?L��A�*

logging/current_cost�"<��#�+       ��K	�mL��A�*

logging/current_cost�*<E� +       ��K	��L��A�*

logging/current_costw8<A�\�+       ��K	��L��A�*

logging/current_costy@<�9�#+       ��K	��L��A�*

logging/current_cost�=<r�'7+       ��K	�%M��A�*

logging/current_cost"6<� �!+       ��K	�RM��A�*

logging/current_cost:<��[�+       ��K	i�M��A�*

logging/current_costlC<�s+       ��K	��M��A�*

logging/current_cost�Y<!��+       ��K	,�M��A�*

logging/current_cost߉<��!+       ��K	�4N��A�*

logging/current_cost#�< �B+       ��K	�fN��A�*

logging/current_costߋ<,�s+       ��K	٤N��A�*

logging/current_cost~<�q*+       ��K	��N��A�*

logging/current_cost��<�p�F+       ��K	�O��A�*

logging/current_costR�<�$�N+       ��K	ZO��A�*

logging/current_cost��<���+       ��K	��O��A�*

logging/current_cost�<��pD+       ��K	��O��A�*

logging/current_cost׮<$��F+       ��K	HP��A�*

logging/current_cost�<-�M*+       ��K	�EP��A�*

logging/current_cost9�<��3+       ��K	��P��A�*

logging/current_cost3�< O`k+       ��K	`�P��A�*

logging/current_cost��<��p�+       ��K	��P��A�*

logging/current_cost�<��+       ��K	;>Q��A�*

logging/current_cost|�<L>�n+       ��K	�uQ��A�*

logging/current_cost��<OE�+       ��K	��Q��A�*

logging/current_cost��<���+       ��K	O�Q��A�*

logging/current_costb�<' �T+       ��K	R��A�*

logging/current_cost��<�*��+       ��K	�;R��A�*

logging/current_costl�<�z�+       ��K	�pR��A�*

logging/current_cost��<,���+       ��K	��R��A�*

logging/current_costvz	<���+       ��K	d�R��A�*

logging/current_cost��	<z�7+       ��K	�S��A�*

logging/current_cost&	<��4?+       ��K	�5S��A�*

logging/current_cost=�<�p��+       ��K	fS��A�*

logging/current_cost��<޽�+       ��K	��S��A�*

logging/current_cost?�<�]�g+       ��K	]�S��A�*

logging/current_cost�;	<�C�C+       ��K	�T��A�*

logging/current_costN
<VH.+       ��K	�1T��A�*

logging/current_cost�k	<{�>�+       ��K	 aT��A�*

logging/current_cost��<+�Z,+       ��K	��T��A�*

logging/current_cost,�<U�-�+       ��K	m�T��A�*

logging/current_cost�+<Օ��+       ��K	��T��A�*

logging/current_cost��<]��8+       ��K	�&U��A�*

logging/current_cost��<쾬+       ��K	�SU��A�*

logging/current_costn�<���+       ��K	��U��A�*

logging/current_cost��<��:�+       ��K	a�U��A�*

logging/current_cost��<�� >+       ��K	��U��A�*

logging/current_cost�<���k+       ��K	�V��A�*

logging/current_costQ�<�f��+       ��K	�AV��A�*

logging/current_cost@�<nʫ�+       ��K	�nV��A�*

logging/current_cost�<3D�+       ��K	�V��A�*

logging/current_costӤ<ih�5+       ��K	��V��A�*

logging/current_cost�<F;��+       ��K	#�V��A�*

logging/current_cost��<�	�+       ��K	�(W��A�*

logging/current_cost��<D�r+       ��K	�UW��A�*

logging/current_costO�<�R�+       ��K	�W��A�*

logging/current_cost��<���+       ��K	��W��A�*

logging/current_cost!s<���,+       ��K	i�W��A�*

logging/current_cost4j<_��+       ��K	7X��A�*

logging/current_costc^<Y|~�+       ��K	�5X��A�*

logging/current_cost�R<��(�+       ��K	1cX��A�*

logging/current_cost�R<����+       ��K	ȐX��A�*

logging/current_cost�D<{�Y+       ��K	��X��A�*

logging/current_cost�?<�}/�+       ��K	y�X��A�*

logging/current_cost%2<��2+       ��K	;Y��A�*

logging/current_costi,<4 ��+       ��K	�MY��A�*

logging/current_costZ"<k��+       ��K	�{Y��A�*

logging/current_cost<�J�F+       ��K	.�Y��A�*

logging/current_cost�<�6o+       ��K	�Y��A�*

logging/current_cost<܀B�+       ��K	�
Z��A�*

logging/current_cost4<���+       ��K	�7Z��A�*

logging/current_costp�<P��+       ��K	0gZ��A�*

logging/current_costy�<d�+       ��K	ēZ��A�*

logging/current_cost��<�S�+       ��K	#�Z��A�*

logging/current_cost
�<E�67+       ��K	t�Z��A�*

logging/current_cost��<H���+       ��K	�.[��A�*

logging/current_cost��<ݓ1+       ��K	J][��A�*

logging/current_cost��<)U+       ��K	N�[��A�*

logging/current_cost�<Hrb�+       ��K	 �[��A�*

logging/current_cost��<9p��+       ��K	��[��A�*

logging/current_cost��<ׁ#+       ��K	\��A�*

logging/current_costa�<�s:B+       ��K	%O\��A�*

logging/current_cost��<�d	+       ��K	;�\��A�*

logging/current_costŲ<��#+       ��K	��\��A�*

logging/current_cost��<Y�+       ��K	��\��A�*

logging/current_cost�<v��Q+       ��K	]��A�*

logging/current_costR�</JӁ+       ��K	�N]��A�*

logging/current_cost��<a��L+       ��K	�}]��A�*

logging/current_cost��<�y��+       ��K	S�]��A�*

logging/current_cost,�<��s�+       ��K	+�]��A�*

logging/current_costD�<�OrV+       ��K	9^��A�*

logging/current_cost�<"=��+       ��K	L^��A�*

logging/current_cost��<��':+       ��K	|^��A�*

logging/current_costU�<в��+       ��K	�^��A�*

logging/current_cost��<�X?�+       ��K	��^��A�*

logging/current_costg�<���#+       ��K	�_��A�*

logging/current_cost��<W%�+       ��K	�<_��A�*

logging/current_cost[�<���+       ��K	m_��A�*

logging/current_costr�<�d�?+       ��K	��_��A�*

logging/current_cost�<s>�b+       ��K	��_��A�*

logging/current_cost	z<PV2�+       ��K	O�_��A�*

logging/current_cost�}<���+       ��K	_$`��A�*

logging/current_cost�y<P!!+       ��K	�Q`��A�*

logging/current_cost�x<��+       ��K	��`��A�*

logging/current_cost�l<b�+       ��K	��`��A�*

logging/current_cost�o<�;�+       ��K	��`��A�*

logging/current_costKi<SG.�+       ��K	da��A�*

logging/current_costpl<�̨�+       ��K	�=a��A�*

logging/current_cost�[<B��<+       ��K	�oa��A�*

logging/current_cost%_<-��+       ��K	��a��A�*

logging/current_cost�S<��Hp+       ��K	��a��A�*

logging/current_cost*L<~@X+       ��K	=�a��A�*

logging/current_cost~V<LdK+       ��K	l#b��A�*

logging/current_cost�T<\�gC+       ��K	�Pb��A�*

logging/current_cost�><>���+       ��K	�}b��A�*

logging/current_costC<߯f+       ��K	Īb��A�*

logging/current_cost�G<�½+       ��K	�b��A�*

logging/current_cost9<x���+       ��K	|c��A�*

logging/current_cost@><��w+       ��K	k3c��A�*

logging/current_costx><���+       ��K	�ac��A�*

logging/current_cost%7<�P�4+       ��K	c��A�*

logging/current_costT)<n��+       ��K	R�c��A�*

logging/current_cost�2<֓x�+       ��K	��c��A�*

logging/current_cost�3<C!)+       ��K	�d��A�*

logging/current_cost )<+\P+       ��K	�Ld��A�*

logging/current_cost�'<W)g�+       ��K	�zd��A�*

logging/current_costB&<���+       ��K	m�d��A�*

logging/current_cost�)<�N��+       ��K	��d��A�*

logging/current_cost�(<0o`+       ��K	�e��A�*

logging/current_cost�<soM+       ��K	�0e��A�*

logging/current_costw(<�+       ��K	x^e��A�*

logging/current_cost�<��JB+       ��K	�e��A�*

logging/current_cost<�!�!+       ��K	��e��A�*

logging/current_cost�<tmr+       ��K	�e��A�*

logging/current_cost6<�W�+       ��K	�f��A�*

logging/current_costG<�Wz+       ��K	�@f��A�*

logging/current_cost�<�Y�1+       ��K	�nf��A�*

logging/current_cost�<�q^+       ��K	G�f��A�*

logging/current_cost�<%,�X+       ��K	��f��A�*

logging/current_cost<�	�+       ��K	�g��A�*

logging/current_cost�<v�j�+       ��K	1g��A�*

logging/current_cost�<.T%�+       ��K	$cg��A�*

logging/current_cost�<*��+       ��K	"�g��A�*

logging/current_costq<� �+       ��K	��g��A�*

logging/current_cost�	<-G�+       ��K	?�g��A�*

logging/current_costE<���+       ��K	}$h��A�*

logging/current_cost�<y�C{+       ��K	6Sh��A�*

logging/current_cost�<�P<�+       ��K	��h��A�*

logging/current_cost� <c#��+       ��K	d�h��A�*

logging/current_cost�<�Rn�+       ��K	��h��A�*

logging/current_cost<)f�k+       ��K	�i��A�*

logging/current_cost�<�Y�`+       ��K	G>i��A�*

logging/current_costt�<��g+       ��K	�li��A�*

logging/current_cost1<���+       ��K	��i��A�*

logging/current_costC�<X�#A+       ��K	��i��A�*

logging/current_costx<��WS+       ��K	��i��A�*

logging/current_cost�<l�h�+       ��K	�'j��A�*

logging/current_cost� <�HI�+       ��K	eVj��A�*

logging/current_cost<(��+       ��K	o�j��A�*

logging/current_cost��<0[;+       ��K	�j��A�*

logging/current_cost� <��"�+       ��K	��j��A�*

logging/current_cost��<͞�&+       ��K	Mk��A�*

logging/current_cost��<�n	+       ��K	�Bk��A�*

logging/current_cost,�<(i7+       ��K	�uk��A�*

logging/current_cost��<y���+       ��K	�k��A�*

logging/current_cost�<ޡ��+       ��K	��k��A�*

logging/current_cost��<��`<+       ��K	(�k��A�*

logging/current_cost��<;��+       ��K	�/l��A�*

logging/current_cost<�<��+       ��K	�]l��A�*

logging/current_cost`�<�'�E+       ��K	��l��A�*

logging/current_costZ�<V�	�+       ��K	�l��A�*

logging/current_costT�<�k�+       ��K	��l��A�*

logging/current_cost��<�@'�+       ��K	�m��A�*

logging/current_costi�<R=2�+       ��K	(Em��A�*

logging/current_costY�<�>Z+       ��K	Mum��A�*

logging/current_cost��<,y�+       ��K	v�m��A�*

logging/current_cost��<S�Z
+       ��K	�m��A�*

logging/current_costl<�٤�+       ��K	\�m��A�*

logging/current_costh�< ��"+       ��K	�,n��A�*

logging/current_cost�<��d+       ��K	�Yn��A�*

logging/current_cost�z<D���+       ��K	��n��A�*

logging/current_cost�e<o�L�+       ��K	1�n��A�*

logging/current_cost,I<m�ݗ+       ��K	��n��A�*

logging/current_costQ9<dN��+       ��K	so��A�*

logging/current_cost4'<]��+       ��K	�:o��A�*

logging/current_cost>$<4�K�+       ��K	&ho��A�*

logging/current_cost�<C}�8+       ��K	��o��A�*

logging/current_cost"<�tc�+       ��K	��o��A�*

logging/current_cost�<5��O+       ��K	F�o��A�*

logging/current_cost)<�2r�+       ��K	�p��A�*

logging/current_cost�<v5��+       ��K	8Gp��A�*

logging/current_cost+<��H+       ��K	^up��A�*

logging/current_cost�<z�#a+       ��K	��p��A�*

logging/current_cost�<WJ	+       ��K	}�p��A�*

logging/current_cost�<��:+       ��K	x�p��A�*

logging/current_cost�<���+       ��K	�,q��A�*

logging/current_cost��<�K@^+       ��K	�Zq��A�*

logging/current_costZ�<͠�I+       ��K	-�q��A�*

logging/current_costk�<�m1�+       ��K	[�q��A�*

logging/current_costQ�<�J��+       ��K	��q��A�*

logging/current_cost��<�1�O+       ��K	&r��A�*

logging/current_cost��<��0,+       ��K	�=r��A�*

logging/current_cost��<�G3�+       ��K	>kr��A�*

logging/current_cost\�<�ן�+       ��K	֘r��A�*

logging/current_cost,�<�3�k+       ��K	��r��A�*

logging/current_cost��<:�+�+       ��K	.�r��A�*

logging/current_cost��<UΥ+       ��K	�#s��A�*

logging/current_cost�<gW��+       ��K	�Qs��A�*

logging/current_costK�<��v�+       ��K	Ks��A�*

logging/current_cost��<��+       ��K	��s��A�*

logging/current_cost��<�7'+       ��K	J�s��A�*

logging/current_cost��<�ی+       ��K	�t��A�*

logging/current_costk�<mlc+       ��K	�/t��A�*

logging/current_cost�<Üg�+       ��K	^t��A�*

logging/current_cost�<�O�H+       ��K	g�t��A�*

logging/current_cost��<����+       ��K	��t��A�*

logging/current_cost��<%�c+       ��K	��t��A�*

logging/current_cost|�<D�+       ��K	
!u��A�*

logging/current_cost��<��5+       ��K	�Nu��A�*

logging/current_cost4�<�ωa+       ��K	|u��A�*

logging/current_cost�<��߶+       ��K	��u��A�*

logging/current_costT�<'I+       ��K	]�u��A�*

logging/current_costr�< ���+       ��K	Qv��A�*

logging/current_costi�<lX��+       ��K	�4v��A�*

logging/current_cost^�<,��d+       ��K	mdv��A�*

logging/current_cost֛<�/��+       ��K	��v��A�*

logging/current_cost��<H��+       ��K	Z�v��A�*

logging/current_cost��<bf��+       ��K	�v��A�*

logging/current_cost<�_<�+       ��K	�"w��A�*

logging/current_cost��<�L9�+       ��K	�Pw��A�*

logging/current_cost��<�e+       ��K	�~w��A�*

logging/current_cost��<�V�Z+       ��K	�w��A�*

logging/current_cost��<�B|�+       ��K	��w��A�*

logging/current_cost��<90+       ��K	�x��A�*

logging/current_cost�r<7�9�+       ��K	�;x��A�*

logging/current_cost�P<����+       ��K	7mx��A�*

logging/current_costR_<mW+       ��K	��x��A�*

logging/current_cost+5<�J`�+       ��K	 �x��A�*

logging/current_cost�U<_��+       ��K	V�x��A�*

logging/current_cost�-<te=4+       ��K	A'y��A�*

logging/current_costcI<�XR�+       ��K	�Xy��A�*

logging/current_cost$7<h��F+       ��K	�y��A�*

logging/current_cost�:<�Ke+       ��K	a�y��A�*

logging/current_cost\H<�Zi�+       ��K	��y��A�*

logging/current_cost�5<r��2+       ��K	�z��A�*

logging/current_costE<��+       ��K	�=z��A�*

logging/current_costND< �@+       ��K	�kz��A�*

logging/current_cost�'<�
0�+       ��K	e�z��A�*

logging/current_costIJ<�0IC+       ��K	w�z��A�*

logging/current_costC*<΢Y�+       ��K	��z��A�*

logging/current_cost�;<[I%/+       ��K	I"{��A�*

logging/current_cost�C<	IH_+       ��K	�M{��A�*

logging/current_cost�!<�M��+       ��K	�{��A�*

logging/current_cost�:< (�>+       ��K	��{��A�*

logging/current_cost+%<�
�7+       ��K	��{��A�*

logging/current_cost�(<P�Q'+       ��K	_|��A�*

logging/current_cost�,<K�*+       ��K	�L|��A�*

logging/current_cost;<�\/R+       ��K	{|��A�*

logging/current_costU<xKt+       ��K	��|��A�*

logging/current_costc@<�[&�+       ��K	�|��A�*

logging/current_cost
2<�l�+       ��K	}��A�*

logging/current_costq$<�$]+       ��K	�.}��A�*

logging/current_costl%<'"+       ��K	�[}��A�*

logging/current_cost�<q9��+       ��K	�}��A�*

logging/current_cost�'<��1�+       ��K	��}��A�*

logging/current_cost��<����+       ��K	x�}��A�*

logging/current_cost/<�c�+       ��K	~��A�*

logging/current_cost�*<F�d�+       ��K	F~��A�*

logging/current_cost��<�ʝ+       ��K	�r~��A�*

logging/current_cost	 <���,+       ��K	��~��A�*

logging/current_costZ<�ɘ�+       ��K	��~��A�*

logging/current_cost�<��9�+       ��K	��~��A�*

logging/current_cost�<���+       ��K	�'��A�*

logging/current_cost�<���;+       ��K	fS��A�*

logging/current_costL<��+       ��K	����A�*

logging/current_cost<��1+       ��K	H���A�*

logging/current_cost�'<N���+       ��K	C���A�*

logging/current_costy	<6+       ��K	����A�*

logging/current_cost��<A��+       ��K	+:���A�*

logging/current_cost�<�N�+       ��K	Lf���A�*

logging/current_cost�<��h+       ��K	����A�*

logging/current_costN�<�5��+       ��K	V����A�*

logging/current_cost��<�;��+       ��K	g�A�*

logging/current_costp+<�Gr�+       ��K	����A�*

logging/current_cost�</*+       ��K	uG���A�*

logging/current_cost�<[4-J+       ��K	4u���A�*

logging/current_cost�5<���+       ��K	u����A�*

logging/current_cost$�<��+       ��K	6ׁ��A�*

logging/current_cost�'<�S.z+       ��K	?���A�*

logging/current_cost��< 8+       ��K	�0���A�*

logging/current_cost@<#2 �+       ��K	'_���A�*

logging/current_costb<�&�+       ��K	�����A�*

logging/current_costo<h��+       ��K	�����A�*

logging/current_cost�<�B+       ��K	�낱�A�*

logging/current_cost��<~U�+       ��K	����A�*

logging/current_cost9<jy+       ��K	8I���A�*

logging/current_cost�<�U�+       ��K	�w���A�*

logging/current_cost��<ҏ�n+       ��K	ܩ���A�*

logging/current_cost�<�7�G+       ��K	0݃��A�*

logging/current_cost�<N�Au+       ��K	����A�*

logging/current_cost*�<t��e+       ��K	�:���A�*

logging/current_costK<ԸG+       ��K	�g���A�*

logging/current_cost&�<ˡ9@+       ��K	�����A�*

logging/current_cost�<��8c+       ��K	�ń��A�*

logging/current_cost\�<�m�W+       ��K	��A�*

logging/current_cost��<�w�+       ��K	�"���A�*

logging/current_cost�<��̵+       ��K	V���A�*

logging/current_cost\<�QEm+       ��K	l����A�*

logging/current_cost�<�*��+       ��K	�����A�*

logging/current_cost��<q'�H+       ��K	�܅��A�*

logging/current_cost�<[�+       ��K	���A�*

logging/current_cost��<�<�+       ��K	<���A�*

logging/current_cost��<1"�H+       ��K	j���A�*

logging/current_cost�<J�]�+       ��K	$����A�*

logging/current_cost��<~+�+       ��K	tǆ��A�*

logging/current_cost��<��6+       ��K	����A�*

logging/current_costD�<EU~+       ��K	&���A�*

logging/current_cost��<�\M�+       ��K	�U���A�*

logging/current_coste�<�l�*+       ��K	3����A�*

logging/current_cost��<�č�+       ��K	�߇��A�*

logging/current_cost��<���+       ��K	R.���A�*

logging/current_cost��<��+       ��K	]y���A�*

logging/current_cost�<f."�+       ��K	J����A�*

logging/current_cost.�<-D�e+       ��K	�����A�*

logging/current_cost+�<��1H+       ��K	a<���A�*

logging/current_cost%�<P�^�+       ��K	�v���A�*

logging/current_cost�<BZ�;+       ��K	Զ���A�*

logging/current_cost%�<����+       ��K	��A�*

logging/current_cost��<i+       ��K	n!���A�*

logging/current_cost��<\|�+       ��K	yZ���A�*

logging/current_cost4�<Z�v�+       ��K	K����A�*

logging/current_costv�<#{�+       ��K	�Ê��A�*

logging/current_costG�<��e+       ��K	����A�*

logging/current_cost��<ҌW]+       ��K	@,���A�*

logging/current_cost��<�U*	+       ��K	�e���A�*

logging/current_cost<�<a�$�+       ��K	�����A�*

logging/current_cost��<4	f+       ��K	�ԋ��A�*

logging/current_cost{�<�%�!+       ��K	����A�*

logging/current_cost6�<����+       ��K	�?���A�*

logging/current_cost��<y�~?+       ��K	�u���A�*

logging/current_costN�<�ԒN+       ��K	�����A�*

logging/current_cost-�<��xX+       ��K	bԌ��A�*

logging/current_cost��<�Pw�+       ��K	5���A�*

logging/current_cost��<��+       ��K	t9���A�*

logging/current_costd�<%D�+       ��K	�g���A�*

logging/current_costS�<�v�+       ��K	����A�*

logging/current_costb�<�5��+       ��K	+ō��A�*

logging/current_cost�<`�+       ��K	���A�*

logging/current_cost�<�:Ĝ+       ��K	N*���A�*

logging/current_costQ�<,��+       ��K	Bf���A�*

logging/current_cost0�<��o+       ��K	7����A�*

logging/current_cost��<�'�+       ��K	Bю��A�*

logging/current_cost��<��+       ��K		���A�*

logging/current_cost��<�gF+       ��K	A/���A�*

logging/current_costu�<㖢+       ��K	]���A�*

logging/current_cost��<��i�+       ��K	�����A�*

logging/current_cost�<�_+       ��K	�����A�*

logging/current_cost��<�$��+       ��K	�叱�A�*

logging/current_cost �<�[+       ��K	����A�*

logging/current_cost��<��Ҡ+       ��K	�D���A�*

logging/current_cost/�<�W�+       ��K	>t���A�*

logging/current_cost��<.7��+       ��K	����A�*

logging/current_cost��<�iG�+       ��K	�ѐ��A�*

logging/current_cost8�<���+       ��K	����A�*

logging/current_costd�<�DG+       ��K	j/���A�*

logging/current_cost<�<�J+       ��K	_^���A�*

logging/current_cost$�<ֱ�+       ��K	�����A�*

logging/current_cost��<�>G0+       ��K	d����A�*

logging/current_cost��<�E�{+       ��K	���A�*

logging/current_costw�<�]�+       ��K	,���A�*

logging/current_cost��<V��/+       ��K	O^���A�*

logging/current_costN�<DNʦ+       ��K	�����A�*

logging/current_cost��<�!c+       ��K	�����A�*

logging/current_cost��< 	)+       ��K	�钱�A�*

logging/current_cost1�<W��+       ��K	����A�*

logging/current_cost�<&k+       ��K	�J���A�*

logging/current_cost��<�&��+       ��K	�x���A�*

logging/current_cost��<{���+       ��K	����A�*

logging/current_cost��<&|Z�+       ��K		ߓ��A�*

logging/current_cost��<�um+       ��K	#���A�*

logging/current_costz�<C�&<+       ��K	�<���A�*

logging/current_cost��<��ԝ+       ��K	*k���A�*

logging/current_cost�<�v��+       ��K	b����A�*

logging/current_cost+�<��+       ��K	S֔��A�*

logging/current_cost~�<�rɂ+       ��K	o���A�*

logging/current_cost2�<��a�+       ��K	94���A�*

logging/current_cost��<g�+       ��K	�a���A�*

logging/current_cost׼<���R+       ��K	�����A�*

logging/current_costE�<�ޡ4+       ��K	Õ��A�*

logging/current_cost|�<�t�t+       ��K	3�A�*

logging/current_cost��<���+       ��K	����A�*

logging/current_cost��<{q�+       ��K	L���A�*

logging/current_cost��<��+       ��K	�z���A�*

logging/current_costR�<�=�+       ��K	����A�*

logging/current_cost9�<�ܰ�+       ��K	�Ֆ��A�*

logging/current_costN�<�vĦ+       ��K	���A�*

logging/current_cost�<ԭ�z+       ��K	�2���A�*

logging/current_cost*�<�R��+       ��K	t_���A�*

logging/current_cost"�<��w+       ��K	�����A�*

logging/current_cost��<�SŘ+       ��K	˸���A�*

logging/current_cost�<��nM+       ��K	�旱�A�*

logging/current_cost��<���+       ��K	����A�*

logging/current_cost��<e��++       ��K	�@���A�*

logging/current_costũ<޾'+       ��K	�n���A�*

logging/current_cost%�<��5+       ��K	h����A�*

logging/current_costԶ<����+       ��K	�͘��A�*

logging/current_cost'�<�:��+       ��K	����A�*

logging/current_costf�<V���+       ��K	t-���A�*

logging/current_cost~�<h�e�+       ��K	9]���A�*

logging/current_cost�<�� :+       ��K	����A�*

logging/current_costԍ<�xT+       ��K	�����A�*

logging/current_costڵ<��+       ��K	�癱�A�*

logging/current_costR�<L�%K+       ��K	:���A�*

logging/current_costL�<X�)�+       ��K	eB���A�*

logging/current_costM�<����+       ��K	q���A�*

logging/current_cost"�<�d'+       ��K	�����A�*

logging/current_cost��<�oMe+       ��K	]̚��A�*

logging/current_costi�<5tG�+       ��K	�����A�*

logging/current_cost��<�g�+       ��K	)+���A�*

logging/current_cost��<�m�+       ��K	jV���A�*

logging/current_cost��<����+       ��K	N����A�*

logging/current_cost��<"Aw+       ��K	����A�*

logging/current_cost^�<}I?+       ��K	K꛱�A�*

logging/current_cost�<��~�+       ��K	\���A�*

logging/current_cost/o<��+       ��K	!I���A�*

logging/current_cost��<r�N�+       ��K	^}���A�*

logging/current_cost֥<�`t�+       ��K	7����A�*

logging/current_cost�o<�3D6+       ��K	֜��A�*

logging/current_costT�<.h��+       ��K	����A�*

logging/current_cost̗<U�!�+       ��K	�9���A�*

logging/current_cost�x<\=5�+       ��K	j���A�*

logging/current_cost�<Nd+       ��K	�����A�*

logging/current_cost�<����+       ��K	�̝��A�*

logging/current_costg�<cn��+       ��K	)����A�*

logging/current_cost��<w���+       ��K	4/���A�*

logging/current_cost{<ӫM�+       ��K	a���A�*

logging/current_cost��<Ƚ�m+       ��K	\����A�*

logging/current_cost��<�=c+       ��K	#����A�*

logging/current_cost��<����+       ��K	Vힱ�A�*

logging/current_cost��<��G+       ��K	����A�*

logging/current_cost�<�)Ho+       ��K	N���A�*

logging/current_cost�<�#�,+       ��K	$}���A�*

logging/current_costpy<�;-9+       ��K	�����A�*

logging/current_costD�<�S=�+       ��K	ן��A�*

logging/current_costG�<s��+       ��K	����A�*

logging/current_costP�<l��+       ��K	�4���A�*

logging/current_costZ�<%��+       ��K	�c���A�*

logging/current_costC�<��$�+       ��K	o����A�*

logging/current_cost.�<��+       ��K	�����A�*

logging/current_cost݋<Y�a�+       ��K	k�A�*

logging/current_cost��<��qC+       ��K	����A�*

logging/current_costz�<�WQ+       ��K	;J���A�*

logging/current_cost)�<����+       ��K	�����A�*

logging/current_costƝ<�(�+       ��K	����A�*

logging/current_cost\�<Go�+       ��K	�ۡ��A�*

logging/current_cost˕<��i;+       ��K	l���A�*

logging/current_cost��<C��Y+       ��K	[4���A�*

logging/current_cost�u<����+       ��K	ba���A�*

logging/current_cost~�<Al�&+       ��K	ԏ���A�*

logging/current_costT�<�D+       ��K	�����A�*

logging/current_cost��<�e}[+       ��K	�뢱�A�*

logging/current_cost]�<��+       ��K	����A�*

logging/current_cost��<x�V+       ��K	:F���A�*

logging/current_cost�v<�H\�+       ��K	�s���A�*

logging/current_cost?�<����+       ��K	ࠣ��A�*

logging/current_costQ�<�pL+       ��K	�Σ��A�*

logging/current_cost��<��֊+       ��K	K����A�*

logging/current_costM�< E�H+       ��K	�)���A�*

logging/current_cost5^<��+       ��K	�W���A�*

logging/current_cost�<LH{�+       ��K	�����A�*

logging/current_cost'r<$�O+       ��K	a����A�*

logging/current_cost�l<��+       ��K	�ᤱ�A�*

logging/current_cost��<��ǘ+       ��K	V���A�*

logging/current_cost7q<	��+       ��K	g?���A�*

logging/current_cost�w<�]Q�+       ��K	�l���A�*

logging/current_cost��<�?�+       ��K	9����A�*

logging/current_cost;l<��L�+       ��K	Xǥ��A� *

logging/current_cost�r<ϫ4�+       ��K	�����A� *

logging/current_cost��<��%`+       ��K	�"���A� *

logging/current_cost�z<�c+       ��K	�O���A� *

logging/current_cost�k<�jn�+       ��K	T���A� *

logging/current_cost��<TqT+       ��K	
����A� *

logging/current_cost$q<1{I�+       ��K	�ߦ��A� *

logging/current_cost�k<l�Xf+       ��K	����A� *

logging/current_cost�<�.��+       ��K	UC���A� *

logging/current_cost\j<rܒ�+       ��K	�p���A� *

logging/current_cost�u<���0+       ��K	�����A� *

logging/current_cost��<���+       ��K	̧��A� *

logging/current_costm}<lWU+       ��K	^����A� *

logging/current_cost�t<���I+       ��K	�*���A� *

logging/current_cost��<�op
+       ��K	Y���A� *

logging/current_costvY<f�*�+       ��K	酨��A� *

logging/current_cost�<�V�{+       ��K	 ����A� *

logging/current_cost��<
�ӭ+       ��K	�쨱�A� *

logging/current_cost�e< ��+       ��K	M���A� *

logging/current_cost	�<צ�+       ��K	cF���A� *

logging/current_cost��<���+       ��K	y���A� *

logging/current_costA[<+7u+       ��K	R����A� *

logging/current_cost��<�z��+       ��K	1թ��A� *

logging/current_cost��<:�+       ��K	+���A� *

logging/current_cost�<^�6#+       ��K	�7���A� *

logging/current_costbl<��g�+       ��K	,h���A�!*

logging/current_costD�<����+       ��K	\����A�!*

logging/current_cost�v<���+       ��K	Ū��A�!*

logging/current_costg�<SƦT+       ��K	6����A�!*

logging/current_cost4m<��+       ��K	�#���A�!*

logging/current_cost�t<��/+       ��K	�Q���A�!*

logging/current_cost��<U�]�+       ��K	��A�!*

logging/current_cost�a<K+       ��K	"����A�!*

logging/current_cost[�<��+       ��K	�ޫ��A�!*

logging/current_costq�<��~+       ��K	����A�!*

logging/current_cost'}<@�+       ��K	�;���A�!*

logging/current_cost�y<��Th+       ��K	l���A�!*

logging/current_cost�|<E��+       ��K	*����A�!*

logging/current_cost�z<G�d~+       ��K	�Ƭ��A�!*

logging/current_costrl<���^+       ��K	�����A�!*

logging/current_cost�s<z:-+       ��K	�#���A�!*

logging/current_costot<,4��+       ��K	�S���A�!*

logging/current_costLt</u/+       ��K	����A�!*

logging/current_costdj<���+       ��K	Q����A�!*

logging/current_cost�{<]b�+       ��K	�ܭ��A�!*

logging/current_cost�f<4)�@+       ��K	g���A�!*

logging/current_cost�l<����+       ��K	#:���A�!*

logging/current_cost��<� B$+       ��K	�u���A�!*

logging/current_cost�g<�:��+       ��K	g����A�!*

logging/current_costAn<�1|+       ��K	sծ��A�!*

logging/current_cost.y<�Ŵ�+       ��K	����A�!*

logging/current_cost�k<��+       ��K	/���A�"*

logging/current_cost�n<�86+       ��K	�\���A�"*

logging/current_cost�r<K!�,+       ��K	G����A�"*

logging/current_cost�k<���+       ��K	路��A�"*

logging/current_cost��<��A+       ��K	�㯱�A�"*

logging/current_coste}<�E$+       ��K	-���A�"*

logging/current_cost�^<�,+?+       ��K	�=���A�"*

logging/current_costEI<����+       ��K	k���A�"*

logging/current_cost��<��M+       ��K	 ����A�"*

logging/current_cost!{<��q+       ��K	RȰ��A�"*

logging/current_cost�M<k,�+       ��K	�����A�"*

logging/current_cost�<��bm+       ��K	�#���A�"*

logging/current_cost2�<�D��+       ��K	#P���A�"*

logging/current_cost^r<ù�+       ��K	�|���A�"*

logging/current_cost�N<z�(I+       ��K	�����A�"*

logging/current_costq<���+       ��K	&ٱ��A�"*

logging/current_cost�j<R�H�+       ��K	h	���A�"*

logging/current_costux<�MZ�+       ��K	7���A�"*

logging/current_costNN<���+       ��K	�d���A�"*

logging/current_costZU<Ibu+       ��K	B����A�"*

logging/current_costd�<��k+       ��K	Ż���A�"*

logging/current_cost�^<N+       ��K	~겱�A�"*

logging/current_cost�Y<u�V+       ��K	����A�"*

logging/current_cost^}<�+       ��K	$F���A�"*

logging/current_costis<v��+       ��K	�r���A�"*

logging/current_costpY<ח�+       ��K	����A�#*

logging/current_cost$B<9�+       ��K	y˳��A�#*

logging/current_cost3g<K�+       ��K	�����A�#*

logging/current_cost��<W��+       ��K	~'���A�#*

logging/current_cost��<W�E++       ��K	eV���A�#*

logging/current_cost&2<GrU+       ��K	����A�#*

logging/current_cost2e<:us+       ��K	̳���A�#*

logging/current_cost�n<%��+       ��K	�ᴱ�A�#*

logging/current_cost||<��w�+       ��K	����A�#*

logging/current_cost�D< ��I+       ��K	O>���A�#*

logging/current_cost�O<M�~+       ��K	�j���A�#*

logging/current_cost s<D|+       ��K	N����A�#*

logging/current_cost�p<��+       ��K	r˵��A�#*

logging/current_cost�_<����+       ��K	�����A�#*

logging/current_cost0Z<BL+       ��K	x(���A�#*

logging/current_cost�j<�$�+       ��K	pU���A�#*

logging/current_cost]<A�0R+       ��K	�����A�#*

logging/current_costd<�+�,+       ��K	�����A�#*

logging/current_cost�G<-l�8+       ��K	H嶱�A�#*

logging/current_costc<��+       ��K	����A�#*

logging/current_cost�e<v�Z�+       ��K	�F���A�#*

logging/current_cost�E<3��+       ��K	'u���A�#*

logging/current_costG=< �e+       ��K	k����A�#*

logging/current_cost�_<0b\+       ��K	�з��A�#*

logging/current_costތ<ߦF+       ��K	����A�#*

logging/current_costLq<޵^+       ��K	D1���A�#*

logging/current_cost�]<�퓽+       ��K	6]���A�$*

logging/current_cost�1<@��&+       ��K	܋���A�$*

logging/current_cost�z<�ޤI+       ��K	0����A�$*

logging/current_cost�<�q��+       ��K	s���A�$*

logging/current_costbS<,�+       ��K	�5���A�$*

logging/current_cost"><U&t�+       ��K	�k���A�$*

logging/current_cost@x<iy��+       ��K	����A�$*

logging/current_cost�q<7��+       ��K	�Ϲ��A�$*

logging/current_cost�j<�|o�+       ��K	���A�$*

logging/current_cost�p<:��y+       ��K	?<���A�$*

logging/current_cost;v<��߂+       ��K	Vr���A�$*

logging/current_costbD<�`��+       ��K	�����A�$*

logging/current_cost�0<)�+       ��K	*غ��A�$*

logging/current_costt<���+       ��K	�	���A�$*

logging/current_cost�V<��p�+       ��K	�A���A�$*

logging/current_costC<4�Ϲ+       ��K	�����A�$*

logging/current_costP<XЁ>+       ��K	�Ի��A�$*

logging/current_cost�I<��G	+       ��K	V���A�$*

logging/current_cost�;<!gr�+       ��K	�f���A�$*

logging/current_cost��<DX�L+       ��K	s����A�$*

logging/current_cost=S<WR�+       ��K	#꼱�A�$*

logging/current_cost�Q<����+       ��K	�)���A�$*

logging/current_cost$o<͌K�+       ��K	�e���A�$*

logging/current_cost+C<�x7�+       ��K	����A�$*

logging/current_cost k<�;ۮ+       ��K	�׽��A�$*

logging/current_costW`<ŌŅ+       ��K	����A�$*

logging/current_cost�M<����+       ��K	BQ���A�%*

logging/current_cost�=<��O+       ��K	P����A�%*

logging/current_costdh<nw7E+       ��K	iھ��A�%*

logging/current_cost�`<�g�+       ��K	����A�%*

logging/current_cost,Y<��7�+       ��K	�B���A�%*

logging/current_cost�^<��"�+       ��K	�x���A�%*

logging/current_cost�S<ط�=+       ��K	�����A�%*

logging/current_cost P<�^=+       ��K	ݿ��A�%*

logging/current_cost[b<�}�+       ��K	����A�%*

logging/current_cost�\<;��+       ��K	�=���A�%*

logging/current_cost�6<�g\�+       ��K	�t���A�%*

logging/current_cost�\<�"+       ��K	]����A�%*

logging/current_cost�t<���+       ��K	&����A�%*

logging/current_costpA<m�+       ��K	����A�%*

logging/current_cost5\<2a+       ��K	(~���A�%*

logging/current_cost|c<a��+       ��K	�����A�%*

logging/current_cost�Q<w�C+       ��K	�±�A�%*

logging/current_costqa<�I�0+       ��K	VV±�A�%*

logging/current_costm]<�I�+       ��K	ݬ±�A�%*

logging/current_cost�F<�9�d+       ��K	��±�A�%*

logging/current_costfO<�Hc�+       ��K	-:ñ�A�%*

logging/current_cost�X<�e^+       ��K	Vqñ�A�%*

logging/current_cost$V<��h+       ��K	v�ñ�A�%*

logging/current_costaM<���+       ��K	$�ñ�A�%*

logging/current_cost�h<A��:+       ��K	(ı�A�%*

logging/current_cost)e<��+W+       ��K	tsı�A�&*

logging/current_costTA<T+       ��K	5�ı�A�&*

logging/current_cost=<�e:Z+       ��K	E�ı�A�&*

logging/current_cost�o<W�[�+       ��K	�0ű�A�&*

logging/current_cost�X<q�d+       ��K	�fű�A�&*

logging/current_costy8<�}��+       ��K	��ű�A�&*

logging/current_costYd<��+       ��K	p�ű�A�&*

logging/current_costEX<�BZ+       ��K	�Ʊ�A�&*

logging/current_costSL<I�+       ��K	�PƱ�A�&*

logging/current_cost�p<Z��+       ��K	F�Ʊ�A�&*

logging/current_cost�b<i�^�+       ��K	��Ʊ�A�&*

logging/current_cost�<f#�+       ��K	��Ʊ�A�&*

logging/current_costES<�7+       ��K	>Ǳ�A�&*

logging/current_costuU<ra�+       ��K	�HǱ�A�&*

logging/current_cost�A<��w+       ��K	xǱ�A�&*

logging/current_cost�[<h��+       ��K	�Ǳ�A�&*

logging/current_costSO<�^�S+       ��K	��Ǳ�A�&*

logging/current_costf<b�tf+       ��K	6ȱ�A�&*

logging/current_cost�I<���+       ��K	�Iȱ�A�&*

logging/current_cost�><�IT]+       ��K	V�ȱ�A�&*

logging/current_cost$A<�e�+       ��K	��ȱ�A�&*

logging/current_costP<Q}��+       ��K	B�ȱ�A�&*

logging/current_costwd<�+       ��K	�ɱ�A�&*

logging/current_cost"I<�j�E+       ��K	�Iɱ�A�&*

logging/current_costK<}7�5+       ��K	�vɱ�A�&*

logging/current_cost�U<p�+       ��K	�ɱ�A�&*

logging/current_cost�=<!K�+       ��K	(�ɱ�A�'*

logging/current_cost�O<�%@j+       ��K	��ɱ�A�'*

logging/current_cost�W<` Ӌ+       ��K	�,ʱ�A�'*

logging/current_cost�P</~31+       ��K	)]ʱ�A�'*

logging/current_costsO<���+       ��K	a�ʱ�A�'*

logging/current_cost�/<:24+       ��K	�ʱ�A�'*

logging/current_cost47<n�"j+       ��K	��ʱ�A�'*

logging/current_costS<8���+       ��K	�˱�A�'*

logging/current_costtH<�
+       ��K	�E˱�A�'*

logging/current_costB6<E# +       ��K	Mw˱�A�'*

logging/current_cost/9<4�1�+       ��K	_�˱�A�'*

logging/current_cost�K<��[$+       ��K	��˱�A�'*

logging/current_cost�y<�;$+       ��K	�̱�A�'*

logging/current_costR<�.�K+       ��K	�0̱�A�'*

logging/current_cost�G<�0�"+       ��K	j^̱�A�'*

logging/current_cost�A<���r+       ��K	��̱�A�'*

logging/current_cost�H<�X+       ��K	]�̱�A�'*

logging/current_costG<��H�+       ��K	^�̱�A�'*

logging/current_cost5<?,�C+       ��K	5ͱ�A�'*

logging/current_cost,9<ۧ��+       ��K	TEͱ�A�'*

logging/current_cost�9<@���+       ��K	/sͱ�A�'*

logging/current_cost�L<On��+       ��K	�ͱ�A�'*

logging/current_cost�N<�$++       ��K	�ͱ�A�'*

logging/current_costb4<[��B+       ��K	<�ͱ�A�'*

logging/current_cost�-<hG�+       ��K	Z2α�A�'*

logging/current_cost�1<���+       ��K	hfα�A�(*

logging/current_cost�b<�8�m+       ��K	͖α�A�(*

logging/current_costta<eusg+       ��K	k�α�A�(*

logging/current_cost><J'�+       ��K	_�α�A�(*

logging/current_cost�9<�~ۮ+       ��K	bϱ�A�(*

logging/current_cost�"<��D�+       ��K	�Qϱ�A�(*

logging/current_cost�P<I-ڬ+       ��K	��ϱ�A�(*

logging/current_cost;h<����+       ��K	v�ϱ�A�(*

logging/current_cost.2< ���+       ��K	a�ϱ�A�(*

logging/current_cost�K<i��+       ��K	�б�A�(*

logging/current_cost[K<���+       ��K	�?б�A�(*

logging/current_cost�8<��E+       ��K	mб�A�(*

logging/current_cost[$<��r+       ��K	��б�A�(*

logging/current_cost�1<��I�+       ��K	3�б�A�(*

logging/current_cost�d<�"s�+       ��K	j�б�A�(*

logging/current_costFV<'���+       ��K	S+ѱ�A�(*

logging/current_cost5<�	�+       ��K	1Zѱ�A�(*

logging/current_cost�4<��>�+       ��K	Ĉѱ�A�(*

logging/current_costUT<w&��+       ��K	ָѱ�A�(*

logging/current_costM<[��q+       ��K	>�ѱ�A�(*

logging/current_cost�1<��+       ��K	�ұ�A�(*

logging/current_cost�/< <+       ��K	�Mұ�A�(*

logging/current_cost�I<=cex+       ��K	��ұ�A�(*

logging/current_cost$]<�6�
+       ��K	_�ұ�A�(*

logging/current_costtF<sD�+       ��K	�ұ�A�(*

logging/current_cost�*<�d��+       ��K	tӱ�A�(*

logging/current_cost<G<$���+       ��K	�?ӱ�A�)*

logging/current_cost�C<b�S+       ��K	Noӱ�A�)*

logging/current_cost�;<)K��+       ��K	��ӱ�A�)*

logging/current_cost�?<,�C+       ��K	>�ӱ�A�)*

logging/current_cost�:<r}j+       ��K	�Ա�A�)*

logging/current_costp.<�s+       ��K	^6Ա�A�)*

logging/current_cost�4<����+       ��K	�cԱ�A�)*

logging/current_cost�#<&�:+       ��K	�Ա�A�)*

logging/current_cost�<<���m+       ��K	��Ա�A�)*

logging/current_cost�/<�&D�+       ��K	r�Ա�A�)*

logging/current_cost�7<L���+       ��K	�ձ�A�)*

logging/current_cost-<�-�+       ��K	:Nձ�A�)*

logging/current_cost16<.���+       ��K	|ձ�A�)*

logging/current_cost^;< ?A+       ��K	��ձ�A�)*

logging/current_cost�?<��V+       ��K	v�ձ�A�)*

logging/current_cost�[<$ � +       ��K	�ֱ�A�)*

logging/current_cost�f<]�m'+       ��K	73ֱ�A�)*

logging/current_cost�><;y�T+       ��K	�bֱ�A�)*

logging/current_costd%<m�պ+       ��K	��ֱ�A�)*

logging/current_cost�2<���o+       ��K	ٿֱ�A�)*

logging/current_cost�><���+       ��K	`�ֱ�A�)*

logging/current_cost><��$ +       ��K	pױ�A�)*

logging/current_costsG<�e�~+       ��K	�Lױ�A�)*

logging/current_cost_<�>2+       ��K	zױ�A�)*

logging/current_cost�H<"��+       ��K	٧ױ�A�)*

logging/current_cost�0<yX�+       ��K	��ױ�A�)*

logging/current_cost#-<.R+       ��K	�ر�A�**

logging/current_cost02<h,��+       ��K	)2ر�A�**

logging/current_cost�V<j�t	+       ��K	�_ر�A�**

logging/current_cost�=<�m�+       ��K	��ر�A�**

logging/current_cost8+<w�$+       ��K	�ر�A�**

logging/current_cost/<l$+       ��K	�ر�A�**

logging/current_cost,Q<!��M+       ��K	�ٱ�A�**

logging/current_costAd<D�r�+       ��K	 Gٱ�A�**

logging/current_cost
O<6�_P+       ��K	ktٱ�A�**

logging/current_cost�@<���+       ��K	�ٱ�A�**

logging/current_cost�'<F%�+       ��K	�ٱ�A�**

logging/current_costt<>/�{+       ��K	��ٱ�A�**

logging/current_costl&<L��+       ��K	�*ڱ�A�**

logging/current_cost	G<��v+       ��K	�Xڱ�A�**

logging/current_costV<8q~�+       ��K	2�ڱ�A�**

logging/current_costd5<ڂ��+       ��K	�ڱ�A�**

logging/current_costx<<̐Q�+       ��K	��ڱ�A�**

logging/current_cost�A<���+       ��K	�۱�A�**

logging/current_costF<��+       ��K	�;۱�A�**

logging/current_cost�G<*�o�+       ��K	�i۱�A�**

logging/current_cost/<S4�+       ��K	t�۱�A�**

logging/current_cost�0<�!�+       ��K	�۱�A�**

logging/current_cost�"<p2+       ��K	��۱�A�**

logging/current_cost<��8�+       ��K	�ܱ�A�**

logging/current_cost�<kƙ�+       ��K	�Qܱ�A�**

logging/current_cost�'<���`+       ��K	�ܱ�A�+*

logging/current_cost72<�G�+       ��K	��ܱ�A�+*

logging/current_cost2S<�+x�+       ��K	 �ܱ�A�+*

logging/current_cost�F<���+       ��K	�ݱ�A�+*

logging/current_cost5<�;�+       ��K	�Dݱ�A�+*

logging/current_cost��<���s+       ��K	urݱ�A�+*

logging/current_cost-<�g&u+       ��K	_�ݱ�A�+*

logging/current_cost�L<��+       ��K	��ݱ�A�+*

logging/current_cost�R<;��+       ��K	{ޱ�A�+*

logging/current_cost�@<7���+       ��K	Z7ޱ�A�+*

logging/current_cost�<<%�d+       ��K	�gޱ�A�+*

logging/current_cost�/<vs�+       ��K	�ޱ�A�+*

logging/current_cost~<�]$+       ��K	��ޱ�A�+*

logging/current_cost�<|�}++       ��K	7�ޱ�A�+*

logging/current_cost�=<��+       ��K	�"߱�A�+*

logging/current_cost\<ƺb_+       ��K	�S߱�A�+*

logging/current_cost�2<�+       ��K	d�߱�A�+*

logging/current_cost�F<Tz�+       ��K	�߱�A�+*

logging/current_costI<ѽ� +       ��K	��߱�A�+*

logging/current_costRW<$���+       ��K	o��A�+*

logging/current_cost�J<��Z(+       ��K	�=��A�+*

logging/current_cost`<B�mG+       ��K	?k��A�+*

logging/current_cost�<�+       ��K	6���A�+*

logging/current_cost�(<�v�+       ��K	����A�+*

logging/current_cost�(<*��+       ��K	����A�+*

logging/current_cost><�<��+       ��K	n-��A�+*

logging/current_costr�<���+       ��K	[[��A�,*

logging/current_cost�G<�v�+       ��K	����A�,*

logging/current_cost�i<��<+       ��K	ַ��A�,*

logging/current_cost�^<�&�+       ��K	���A�,*

logging/current_cost�<;��+       ��K	c��A�,*

logging/current_cost��<z���+       ��K	`A��A�,*

logging/current_cost~N<Ì�+       ��K	�o��A�,*

logging/current_cost�{<6c�j+       ��K	���A�,*

logging/current_cost�< ��R+       ��K	����A�,*

logging/current_cost"<Q�e�+       ��K	y���A�,*

logging/current_costIN<�M+       ��K	�&��A�,*

logging/current_cost�P<�(�g+       ��K	T��A�,*

logging/current_cost H<6��u+       ��K	����A�,*

logging/current_cost�P<�kfc+       ��K	u���A�,*

logging/current_cost�[<'9Aw+       ��K	���A�,*

logging/current_cost�=<
�,�+       ��K	��A�,*

logging/current_cost{8<_�~+       ��K	�?��A�,*

logging/current_cost�@<����+       ��K	Cm��A�,*

logging/current_cost)x<���]+       ��K	3���A�,*

logging/current_cost�:<��>+       ��K	f���A�,*

logging/current_cost��<�~+       ��K	1���A�,*

logging/current_cost�?<���+       ��K	$$��A�,*

logging/current_cost`y<\��+       ��K	lP��A�,*

logging/current_cost�-<�]!+       ��K	��A�,*

logging/current_cost� <	E��+       ��K	���A�,*

logging/current_costh=<d�h+       ��K	����A�-*

logging/current_cost,f<l�+       ��K	h��A�-*

logging/current_cost�2<j��+       ��K	5��A�-*

logging/current_cost�<{M�+       ��K	�b��A�-*

logging/current_cost�><�{+       ��K	~���A�-*

logging/current_cost,`<�R��+       ��K	#���A�-*

logging/current_cost�+<(��+       ��K	���A�-*

logging/current_cost�<{��+       ��K	^'��A�-*

logging/current_cost�<+��+       ��K	�U��A�-*

logging/current_cost�0<���+       ��K	���A�-*

logging/current_cost<U<p�%�+       ��K	
���A�-*

logging/current_cost!S<4� +       ��K	����A�-*

logging/current_costu <yh�+       ��K	8��A�-*

logging/current_cost�<���+       ��K	�6��A�-*

logging/current_cost� <L��+       ��K	�d��A�-*

logging/current_costD^<|>s+       ��K	���A�-*

logging/current_cost�<�N�+       ��K	f���A�-*

logging/current_cost�<��3+       ��K	_���A�-*

logging/current_cost�,<���Y+       ��K	���A�-*

logging/current_costZS<Ò��+       ��K	uK��A�-*

logging/current_costd!<����+       ��K	�}��A�-*

logging/current_cost�<��k+       ��K	����A�-*

logging/current_cost�/<ŕ��+       ��K	����A�-*

logging/current_costLV<B�q�+       ��K	O��A�-*

logging/current_costf<�u��+       ��K	\5��A�-*

logging/current_cost�<��vG+       ��K	�d��A�-*

logging/current_cost�)<��}%+       ��K	o���A�.*

logging/current_cost�O<�3+       ��K	����A�.*

logging/current_cost<<"��+       ��K	����A�.*

logging/current_cost><�l�5+       ��K	�!��A�.*

logging/current_cost�,<fN)+       ��K	�P��A�.*

logging/current_costrH<��+       ��K	���A�.*

logging/current_cost&N<�>m�+       ��K	԰��A�.*

logging/current_cost�<p�2C+       ��K	5���A�.*

logging/current_cost�<OU�+       ��K	��A�.*

logging/current_cost�U<u���+       ��K	s;��A�.*

logging/current_cost�B<T��F+       ��K	Jn��A�.*

logging/current_cost�<���B+       ��K	G���A�.*

logging/current_cost9<
��+       ��K	����A�.*

logging/current_cost�Y<q��
+       ��K	M���A�.*

logging/current_costkD<4���+       ��K	7(���A�.*

logging/current_cost�<� Uo+       ��K	�W���A�.*

logging/current_costT<&��+       ��K	 ����A�.*

logging/current_cost<W<��Q�+       ��K	�����A�.*

logging/current_cost�F<'�� +       ��K	u����A�.*

logging/current_costu-<�C+       ��K	���A�.*

logging/current_cost`<#�i�+       ��K	!D��A�.*

logging/current_cost-9<Ʃ�+       ��K	2s��A�.*

logging/current_cost�]<�MV�+       ��K	����A�.*

logging/current_cost\$<Y3��+       ��K	����A�.*

logging/current_cost�<�N�r+       ��K	A���A�.*

logging/current_cost�2<z[z�+       ��K	�,��A�.*

logging/current_cost�S<~1�+       ��K	�]��A�/*

logging/current_cost�+<��"(+       ��K	����A�/*

logging/current_cost�<p"�:+       ��K	Ż��A�/*

logging/current_cost�9<hd�+       ��K	U���A�/*

logging/current_cost^V<	g�+       ��K	��A�/*

logging/current_cost�,<,�ޞ+       ��K	J��A�/*

logging/current_cost<g[�+       ��K	<x��A�/*

logging/current_cost�G<|���+       ��K	����A�/*

logging/current_costE6<H�$�+       ��K	m���A�/*

logging/current_cost�
<\��W+       ��K	����A�/*

logging/current_costNa<Eީg+       ��K	-��A�/*

logging/current_cost�E<W:�Q+       ��K	�Z��A�/*

logging/current_cost�<`��j+       ��K	z���A�/*

logging/current_costnH<qR)�+       ��K	���A�/*

logging/current_cost�*<�f�Q+       ��K	����A�/*

logging/current_cost�<��u+       ��K	��A�/*

logging/current_cost�.<2���+       ��K	�<��A�/*

logging/current_cost-.<{ٔf+       ��K	�j��A�/*

logging/current_cost� <����+       ��K	ݖ��A�/*

logging/current_cost�0<���+       ��K	����A�/*

logging/current_cost�.<�8L�+       ��K	0���A�/*

logging/current_cost{$<V$A+       ��K	���A�/*

logging/current_cost�8<����+       ��K	_K��A�/*

logging/current_cost�9<��0p+       ��K	!x��A�/*

logging/current_cost�<�G��+       ��K	����A�/*

logging/current_cost�<�C_�+       ��K	P���A�0*

logging/current_coste<�+       ��K	����A�0*

logging/current_cost�2<
p��+       ��K	�.���A�0*

logging/current_cost2<�W�+       ��K	P\���A�0*

logging/current_costl,<8)�+       ��K	����A�0*

logging/current_costu�<#�g+       ��K	�����A�0*

logging/current_costKM<�xp�+       ��K	�����A�0*

logging/current_cost�C<!�h�+       ��K	����A�0*

logging/current_cost�<r?l�+       ��K	)A���A�0*

logging/current_costw0<8��5+       ��K	m���A�0*

logging/current_cost_B<l�+       ��K	{����A�0*

logging/current_cost�<LJr+       ��K	r����A�0*

logging/current_cost�!<A�p+       ��K	����A�0*

logging/current_cost�4<���+       ��K	4!���A�0*

logging/current_cost�%<��+       ��K	N���A�0*

logging/current_cost�+<��+       ��K	�~���A�0*

logging/current_cost�<�Ł+       ��K	\����A�0*

logging/current_cost		</��+       ��K	O����A�0*

logging/current_cost><B6n+       ��K	m���A�0*

logging/current_cost-+<�w{+       ��K	O:���A�0*

logging/current_costE<�W� +       ��K	f���A�0*

logging/current_costt6<"g+       ��K	6����A�0*

logging/current_cost�(<[ `�+       ��K	T����A�0*

logging/current_cost�E<3U�+       ��K	����A�0*

logging/current_cost�I<-�y�+       ��K	�!���A�0*

logging/current_costQ&<s��+       ��K	R���A�0*

logging/current_cost�<���+       ��K	����A�1*

logging/current_costL0<I�Fp+       ��K	����A�1*

logging/current_costGW<���+       ��K	�����A�1*

logging/current_costv$<���k+       ��K	����A�1*

logging/current_cost><����+       ��K	�:���A�1*

logging/current_cost�<���G+       ��K	�g���A�1*

logging/current_cost�8<U�+       ��K	�����A�1*

logging/current_cost�H<�FgW+       ��K	S����A�1*

logging/current_cost4:<�~G�+       ��K	����A�1*

logging/current_cost�<��V�+       ��K	m ���A�1*

logging/current_cost	<H�b�+       ��K	�O���A�1*

logging/current_costdY<'�&�+       ��K	E|���A�1*

logging/current_costb4<
7��+       ��K	Ϫ���A�1*

logging/current_cost�<��a+       ��K	�����A�1*

logging/current_cost�%<;��g+       ��K	����A�1*

logging/current_cost5U<$ub�+       ��K	�4���A�1*

logging/current_costG5<2��+       ��K	Tb���A�1*

logging/current_costk<��+       ��K	n����A�1*

logging/current_cost�&<��n`+       ��K	����A�1*

logging/current_costS<K�y+       ��K	�A���A�1*

logging/current_cost9<;:|�+       ��K	6����A�1*

logging/current_cost�<����+       ��K	�����A�1*

logging/current_cost�<�+       ��K	o����A�1*

logging/current_cost�<�3��+       ��K	(&���A�1*

logging/current_costK<����+       ��K	?V���A�1*

logging/current_cost�G<����+       ��K	ی���A�2*

logging/current_cost`<|��,+       ��K	�����A�2*

logging/current_cost</(,++       ��K	�����A�2*

logging/current_costrW<�G)+       ��K	�,���A�2*

logging/current_cost�!<��ؽ+       ��K	�^���A�2*

logging/current_cost��<|���+       ��K	����A�2*

logging/current_costlL<*��+       ��K	����A�2*

logging/current_cost�2<?�܌+       ��K	�����A�2*

logging/current_cost)<�@c+       ��K		���A�2*

logging/current_costyY<w�+       ��K	�O���A�2*

logging/current_costa5<�V��+       ��K	~���A�2*

logging/current_cost��<��+       ��K	�����A�2*

logging/current_cost22<l���+       ��K	>����A�2*

logging/current_cost�G<2�oa+       ��K	� ��A�2*

logging/current_costs<��s�+       ��K	h4 ��A�2*

logging/current_cost<,�(�+       ��K	c ��A�2*

logging/current_cost_<A�+�+       ��K	� ��A�2*

logging/current_cost; <�}�+       ��K	�� ��A�2*

logging/current_cost<	�ƌ+       ��K	�� ��A�2*

logging/current_costX8<gy�/+       ��K	���A�2*

logging/current_costj+<9�Ȣ+       ��K	J��A�2*

logging/current_cost�7<�� +       ��K	.{��A�2*

logging/current_cost<��<+       ��K	r���A�2*

logging/current_cost.<���~+       ��K	����A�2*

logging/current_costVK<�l�+       ��K	g	��A�2*

logging/current_cost��<z���+       ��K	�>��A�2*

logging/current_costW<1|�/+       ��K	�n��A�3*

logging/current_costK7<��T+       ��K	����A�3*

logging/current_costG<2��|+       ��K	����A�3*

logging/current_cost�<��-Y+       ��K	����A�3*

logging/current_cost	L<�@��+       ��K	�%��A�3*

logging/current_cost�< m�+       ��K	�T��A�3*

logging/current_cost�<���)+       ��K	_���A�3*

logging/current_costZ<3�|�+       ��K	����A�3*

logging/current_cost�<\��+       ��K	-���A�3*

logging/current_cost5<�}� +       ��K	4��A�3*

logging/current_costr<��y�+       ��K	ED��A�3*

logging/current_costF<_�@+       ��K	q��A�3*

logging/current_costQ<���u+       ��K	����A�3*

logging/current_cost�N<.]i+       ��K	%���A�3*

logging/current_costE<^��+       ��K	����A�3*

logging/current_cost<$��c+       ��K	1��A�3*

logging/current_costE'<c��+       ��K	�a��A�3*

logging/current_cost�P<'�?+       ��K	v���A�3*

logging/current_costq<m��U+       ��K	����A�3*

logging/current_cost�<�Z��+       ��K	����A�3*

logging/current_cost�4<H-�{+       ��K	���A�3*

logging/current_cost�J<7O�+       ��K	>K��A�3*

logging/current_costa!<�y9+       ��K	�x��A�3*

logging/current_cost�<;V��+       ��K	k���A�3*

logging/current_costU[<(�U}+       ��K	����A�3*

logging/current_cost	<�Q��+       ��K	��A�3*

logging/current_cost*<9��1+       ��K	66��A�4*

logging/current_cost�[<��J�+       ��K	"f��A�4*

logging/current_cost��<<�S�+       ��K	:���A�4*

logging/current_cost�<�gk+       ��K	z���A�4*

logging/current_cost�7<-b��+       ��K	����A�4*

logging/current_cost�7<lo/�+       ��K	���A�4*

logging/current_costw�<#y��+       ��K	[N��A�4*

logging/current_costQ.<�/�+       ��K	S���A�4*

logging/current_cost�E<P1��+       ��K	D���A�4*

logging/current_cost�<��J+       ��K	A���A�4*

logging/current_cost#<��(�+       ��K	�	��A�4*

logging/current_cost�Z<�JQ+       ��K	4?	��A�4*

logging/current_cost<��+       ��K	to	��A�4*

logging/current_cost�<���+       ��K	��	��A�4*

logging/current_cost�T<��.d+       ��K	��	��A�4*

logging/current_cost�<�$w�+       ��K	��	��A�4*

logging/current_cost�<G;uB+       ��K	(
��A�4*

logging/current_cost�N<F���+       ��K	�W
��A�4*

logging/current_costW <���++       ��K	�
��A�4*

logging/current_cost<�MQ+       ��K	�
��A�4*

logging/current_costI$<��X.+       ��K	l�
��A�4*

logging/current_cost�5<��+       ��K	��A�4*

logging/current_cost�<X�&+       ��K	>��A�4*

logging/current_cost5<���D+       ��K	n��A�4*

logging/current_cost�<Y��%+       ��K	ɛ��A�4*

logging/current_cost�5<
��B+       ��K	����A�5*

logging/current_cost�<��;�+       ��K	����A�5*

logging/current_costC<3�@+       ��K	�$��A�5*

logging/current_cost�1< E\�+       ��K	�R��A�5*

logging/current_costC<�
ƺ+       ��K	����A�5*

logging/current_cost7<�h�%+       ��K	���A�5*

logging/current_cost-<ni|\+       ��K	���A�5*

logging/current_cost�<T3+       ��K	��A�5*

logging/current_cost�J<���+       ��K	i5��A�5*

logging/current_cost-9<_�J�+       ��K	xb��A�5*

logging/current_costn<K25�+       ��K	����A�5*

logging/current_costn<�Zc+       ��K	����A�5*

logging/current_cost^C<;x�+       ��K	����A�5*

logging/current_costt3<L��O+       ��K	�1��A�5*

logging/current_cost<���+       ��K	�y��A�5*

logging/current_costvZ<�5+       ��K	���A�5*

logging/current_costN)<3�?�+       ��K	����A�5*

logging/current_cost�<�>+       ��K	�*��A�5*

logging/current_costLQ<�h�k+       ��K	jq��A�5*

logging/current_cost -<�j�;+       ��K	G���A�5*

logging/current_cost��<�e,+       ��K	����A�5*

logging/current_cost�'<}Q�+       ��K	�)��A�5*

logging/current_cost�@<��C+       ��K	T\��A�5*

logging/current_costR<O�E�+       ��K	����A�5*

logging/current_cost�<17�+       ��K	���A�5*

logging/current_cost8<f�h�+       ��K	��A�5*

logging/current_cost(0<~���+       ��K	�@��A�6*

logging/current_cost^<Јo?+       ��K	�t��A�6*

logging/current_cost�<3�2�+       ��K	0���A�6*

logging/current_cost2<����+       ��K	<���A�6*

logging/current_cost�4<�5��+       ��K	�$��A�6*

logging/current_cost��<� 1�+       ��K	`��A�6*

logging/current_cost�;<�i+       ��K	R���A�6*

logging/current_cost,6<�^��+       ��K	���A�6*

logging/current_costE�< ���+       ��K	����A�6*

logging/current_costZ<���+       ��K	�%��A�6*

logging/current_cost6<�69�+       ��K	�S��A�6*

logging/current_costI<2��c+       ��K	����A�6*

logging/current_cost�<<�}+       ��K	����A�6*

logging/current_cost'<�G��+       ��K	����A�6*

logging/current_costu<���+       ��K	���A�6*

logging/current_cost*<{��+       ��K	�G��A�6*

logging/current_cost\-<5~ɼ+       ��K	$x��A�6*

logging/current_costJ<�s+       ��K	����A�6*

logging/current_costI<w픏+       ��K	����A�6*

logging/current_cost�<g +       ��K	e"��A�6*

logging/current_cost <��+       ��K	O��A�6*

logging/current_costU<<@�"�+       ��K	�z��A�6*

logging/current_cost�!<�	i�+       ��K	���A�6*

logging/current_cost+"<d~?+       ��K	����A�6*

logging/current_cost�<�.+       ��K	���A�6*

logging/current_costk <&~�~+       ��K		B��A�7*

logging/current_costX!<���+       ��K	�l��A�7*

logging/current_cost�4<E���+       ��K	W���A�7*

logging/current_costu<�ڞ�+       ��K	%���A�7*

logging/current_cost�*<�1Z�+       ��K	o���A�7*

logging/current_costp<\m�+       ��K	�*��A�7*

logging/current_costk <�ׯu+       ��K	X��A�7*

logging/current_cost�*<}�$�+       ��K	^���A�7*

logging/current_cost(<)�+       ��K	����A�7*

logging/current_cost�'<�	O�+       ��K	����A�7*

logging/current_cost E<���+       ��K	j��A�7*

logging/current_cost�<��m�+       ��K	V=��A�7*

logging/current_cost�)<��SA+       ��K	�i��A�7*

logging/current_costH<���+       ��K	y���A�7*

logging/current_cost�<�2+       ��K	����A�7*

logging/current_cost�2<w���+       ��K	����A�7*

logging/current_costtQ<�`-:+       ��K	o!��A�7*

logging/current_cost�<���+       ��K	Q��A�7*

logging/current_cost�<�F+       ��K	5��A�7*

logging/current_cost,<�H�A+       ��K	����A�7*

logging/current_cost�V<�>+       ��K	����A�7*

logging/current_cost�<,AG+       ��K	���A�7*

logging/current_cost7�<�(�a+       ��K	]8��A�7*

logging/current_cost<<�^��+       ��K	�g��A�7*

logging/current_cost%-<=�c�+       ��K	����A�7*

logging/current_cost�C<!wb�+       ��K	���A�7*

logging/current_cost5<k��p+       ��K	����A�8*

logging/current_cost�<����+       ��K	�1��A�8*

logging/current_cost�<i�x+       ��K	"a��A�8*

logging/current_cost�Q<�R�j+       ��K	1���A�8*

logging/current_cost�3<�G�l+       ��K	���A�8*

logging/current_cost�<JE$+       ��K	z���A�8*

logging/current_cost�<31�+       ��K	
��A�8*

logging/current_cost�O<��{�+       ��K	%I��A�8*

logging/current_costD<ÿe�+       ��K	�x��A�8*

logging/current_cost�<s�	+       ��K	���A�8*

logging/current_costM<"C�+       ��K	7���A�8*

logging/current_cost,.<]Y�+       ��K	���A�8*

logging/current_costv
<g/�@+       ��K	$5��A�8*

logging/current_costdC<��+       ��K	�c��A�8*

logging/current_costn'<�ok�+       ��K	����A�8*

logging/current_cost��<�-6�+       ��K	n���A�8*

logging/current_cost�T<"�f�+       ��K	����A�8*

logging/current_cost+0<����+       ��K	"-��A�8*

logging/current_cost�<��(+       ��K	6\��A�8*

logging/current_cost\7<d�1p+       ��K	���A�8*

logging/current_cost9<�e[+       ��K	����A�8*

logging/current_cost><��f+       ��K	w���A�8*

logging/current_cost7!<�\��+       ��K	��A�8*

logging/current_cost�<L���+       ��K	�F��A�8*

logging/current_coste(<���O+       ��K	5w��A�8*

logging/current_cost�?<��A+       ��K	v���A�8*

logging/current_cost�,<�Fs�+       ��K	����A�9*

logging/current_cost;�<��"�+       ��K	Z ��A�9*

logging/current_costw<\~��+       ��K	g6 ��A�9*

logging/current_costM*<�=�<+       ��K	�i ��A�9*

logging/current_cost?<(�ƚ+       ��K	;� ��A�9*

logging/current_cost)3<F,�p+       ��K	U� ��A�9*

logging/current_cost�<'lKS+       ��K	�� ��A�9*

logging/current_cost�<=�F+       ��K	U'!��A�9*

logging/current_costM<S�+       ��K	�T!��A�9*

logging/current_cost*<$;$�+       ��K	х!��A�9*

logging/current_cost�<���,+       ��K	ʳ!��A�9*

logging/current_cost{ <�Ra�+       ��K	��!��A�9*

logging/current_cost�@<ճ��+       ��K	G"��A�9*

logging/current_costn<��~o+       ��K	�?"��A�9*

logging/current_cost/�<��i+       ��K	Lm"��A�9*

logging/current_costg)<S Z�+       ��K	�"��A�9*

logging/current_cost
<<!���+       ��K	��"��A�9*

logging/current_cost�6<0A�+       ��K	��"��A�9*

logging/current_cost�"<��K+       ��K	'#��A�9*

logging/current_cost��<��I�+       ��K	9T#��A�9*

logging/current_cost"<�[\+       ��K	��#��A�9*

logging/current_cost�8<`Ǹ+       ��K	��#��A�9*

logging/current_cost@<�	�+       ��K	m�#��A�9*

logging/current_cost�<�gpA+       ��K	�$��A�9*

logging/current_cost�	< n,+       ��K	d<$��A�9*

logging/current_cost�S<c[�*+       ��K	th$��A�:*

logging/current_costH2<�X@+       ��K	�$��A�:*

logging/current_cost�<n	N�+       ��K	��$��A�:*

logging/current_cost�<"[/�+       ��K	d�$��A�:*

logging/current_costYH<��$�+       ��K	;!%��A�:*

logging/current_cost\5<�\�@+       ��K	Q%��A�:*

logging/current_cost�%<֓#�+       ��K	�~%��A�:*

logging/current_cost) <$�5S+       ��K	��%��A�:*

logging/current_cost�<p���+       ��K	`�%��A�:*

logging/current_cost�1<�Ʉ�+       ��K	�
&��A�:*

logging/current_costh;<N�M+       ��K	.8&��A�:*

logging/current_cost��<ǭr�+       ��K	jf&��A�:*

logging/current_costP<��+       ��K	D�&��A�:*

logging/current_cost�5<�j*m+       ��K	j�&��A�:*

logging/current_cost	-<*�W+       ��K	��&��A�:*

logging/current_cost@<��+       ��K	�('��A�:*

logging/current_cost6<���R+       ��K	hV'��A�:*

logging/current_costrL<4H�}+       ��K	��'��A�:*

logging/current_costT-<���+       ��K	z�'��A�:*

logging/current_cost�<wH;+       ��K	^�'��A�:*

logging/current_cost7><��"�+       ��K	V(��A�:*

logging/current_cost�.<��c+       ��K	�<(��A�:*

logging/current_cost��<I�7�+       ��K	l(��A�:*

logging/current_cost�=<���+       ��K	+�(��A�:*

logging/current_cost�:<y�:-+       ��K	��(��A�:*

logging/current_cost	<�{+       ��K	l�(��A�:*

logging/current_cost�	<ֲ�+       ��K	5!)��A�;*

logging/current_cost�K<C�c�+       ��K	�N)��A�;*

logging/current_cost3<���+       ��K	�})��A�;*

logging/current_cost,#<{���+       ��K	}�)��A�;*

logging/current_costo�<��+       ��K	��)��A�;*

logging/current_cost~<`=i+       ��K	�*��A�;*

logging/current_cost�2<x�b+       ��K	�4*��A�;*

logging/current_costmI<;���+       ��K	�b*��A�;*

logging/current_cost�<��+       ��K	�*��A�;*

logging/current_cost�<%}?�+       ��K	��*��A�;*

logging/current_cost�6<���G+       ��K	i�*��A�;*

logging/current_costW4<�Z +       ��K	
&+��A�;*

logging/current_cost��<Ȧ�+       ��K	�U+��A�;*

logging/current_cost5<�=��+       ��K	G�+��A�;*

logging/current_costT7<x��+       ��K	��+��A�;*

logging/current_cost�/<Q�Y�+       ��K	��+��A�;*

logging/current_cost��<ot�+       ��K	f,��A�;*

logging/current_costR3<;j-�+       ��K	�;,��A�;*

logging/current_cost�?<K;��+       ��K	*n,��A�;*

logging/current_cost��<�e��+       ��K	�,��A�;*

logging/current_cost�V<�kd�+       ��K	@�,��A�;*

logging/current_cost&4<tR�i+       ��K	��,��A�;*

logging/current_cost��<ej+       ��K	�,-��A�;*

logging/current_cost�3<1ئF+       ��K	B^-��A�;*

logging/current_cost?@<�{z�+       ��K	��-��A�;*

logging/current_cost<��m�+       ��K	��-��A�<*

logging/current_cost*�<�n�=+       ��K		�-��A�<*

logging/current_cost� <��<�+       ��K	�.��A�<*

logging/current_cost�:<h�u+       ��K	�E.��A�<*

logging/current_cost^<N���+       ��K	�s.��A�<*

logging/current_costl�<�x'+       ��K	�.��A�<*

logging/current_cost>a<�z��+       ��K	��.��A�<*

logging/current_cost�#<�C�,+       ��K	/��A�<*

logging/current_cost��<�z��+       ��K	R-/��A�<*

logging/current_cost<<��-�+       ��K	t]/��A�<*

logging/current_cost<����+       ��K	�/��A�<*

logging/current_cost��<�T�X+       ��K	]�/��A�<*

logging/current_costC<�d��+       ��K	.�/��A�<*

logging/current_cost�7<��&I+       ��K	�0��A�<*

logging/current_cost�	<Dv~+       ��K	G0��A�<*

logging/current_costP6<�S��+       ��K	�u0��A�<*

logging/current_cost<���	+       ��K	��0��A�<*

logging/current_cost&(<�?�K+       ��K	m�0��A�<*

logging/current_cost�,<�ԡ+       ��K	1��A�<*

logging/current_cost�<��R1+       ��K	</1��A�<*

logging/current_costY <%��+       ��K	�\1��A�<*

logging/current_cost5#<�5w�+       ��K	��1��A�<*

logging/current_cost�<���+       ��K	*�1��A�<*

logging/current_cost?<;m@�+       ��K	�1��A�<*

logging/current_cost�<X���+       ��K	�2��A�<*

logging/current_cost�A<»�+       ��K	2D2��A�<*

logging/current_costk:<����+       ��K	<q2��A�=*

logging/current_cost<E���+       ��K	2�2��A�=*

logging/current_costt<-�e�+       ��K	e�2��A�=*

logging/current_cost�<���>+       ��K	��2��A�=*

logging/current_cost�?<���g+       ��K	�"3��A�=*

logging/current_cost�:<EGl�+       ��K	[Q3��A�=*

logging/current_costu<���m+       ��K	�~3��A�=*

logging/current_cost�<�b�+       ��K	�3��A�=*

logging/current_cost�G<��7�+       ��K	��3��A�=*

logging/current_cost|2<Z�>�+       ��K	�4��A�=*

logging/current_cost` <����+       ��K	�44��A�=*

logging/current_costT<��G8+       ��K	|a4��A�=*

logging/current_cost�K<��b+       ��K	�4��A�=*

logging/current_cost94<1���+       ��K	e�4��A�=*

logging/current_cost<�H+       ��K	n�4��A�=*

logging/current_costV<[�`+       ��K	:5��A�=*

logging/current_cost�I<�	!#+       ��K	�D5��A�=*

logging/current_cost�2<��M)+       ��K	�r5��A�=*

logging/current_cost�'<�Uت+       ��K	ş5��A�=*

logging/current_cost5�<��+       ��K	;�5��A�=*

logging/current_cost�<e���+       ��K	�5��A�=*

logging/current_cost�+<� +       ��K	�%6��A�=*

logging/current_cost^7<�,��+       ��K	IW6��A�=*

logging/current_cost��<���X+       ��K	��6��A�=*

logging/current_costl<��P[+       ��K	��6��A�=*

logging/current_cost�.<+x�+       ��K	��6��A�=*

logging/current_costZ=<DU+       ��K	�7��A�>*

logging/current_costX<�AG�+       ��K	�97��A�>*

logging/current_cost�<��s�+       ��K	Ee7��A�>*

logging/current_costW2<��<+       ��K	ے7��A�>*

logging/current_cost_5<�;{�+       ��K	þ7��A�>*

logging/current_cost$�<�+       ��K	��7��A�>*

logging/current_cost<�VG+       ��K	�8��A�>*

logging/current_cost�/<J++       ��K	�N8��A�>*

logging/current_cost�0<�ݷF+       ��K	�{8��A�>*

logging/current_cost�<!�y�+       ��K	��8��A�>*

logging/current_cost�<��v+       ��K	�8��A�>*

logging/current_costB*<g�4+       ��K	K9��A�>*

logging/current_cost*<w`�+       ��K	�69��A�>*

logging/current_costv<��1�+       ��K	�c9��A�>*

logging/current_cost�<��t�+       ��K	�9��A�>*

logging/current_costPR<H83�+       ��K	��9��A�>*

logging/current_cost<�A�+       ��K	��9��A�>*

logging/current_coste <'�)�+       ��K	�":��A�>*

logging/current_cost�<)�!�+       ��K	�V:��A�>*

logging/current_cost�"<G�9+       ��K	G�:��A�>*

logging/current_cost�A<�M�+       ��K	�:��A�>*

logging/current_costS1<�$��+       ��K	��:��A�>*

logging/current_cost�<B���+       ��K	T;��A�>*

logging/current_cost�<  D+       ��K	x<;��A�>*

logging/current_cost�< B�+       ��K	"w;��A�>*

logging/current_costU<5��<+       ��K	��;��A�?*

logging/current_cost�'<���+       ��K	.<��A�?*

logging/current_cost<.��+       ��K	N<��A�?*

logging/current_costU!<-���+       ��K	֥<��A�?*

logging/current_cost�<V�q+       ��K	��<��A�?*

logging/current_cost�%<�� v+       ��K	�=��A�?*

logging/current_cost�<a�t�+       ��K	�F=��A�?*

logging/current_costu(<���+       ��K	ŕ=��A�?*

logging/current_cost�<���+       ��K	��=��A�?*

logging/current_costhE<2d��+       ��K	n>��A�?*

logging/current_cost'<���<+       ��K	<>��A�?*

logging/current_cost,<�(XR+       ��K	�t>��A�?*

logging/current_cost�8<����+       ��K	�>��A�?*

logging/current_cost�"<l¨+       ��K	Y�>��A�?*

logging/current_costn.<�U�T+       ��K	?��A�?*

logging/current_cost�,<��a+       ��K	�1?��A�?*

logging/current_cost�*<�SP+       ��K	�g?��A�?*

logging/current_cost�	<,4]^+       ��K	�?��A�?*

logging/current_cost��<��%�+       ��K	��?��A�?*

logging/current_cost�0<��A+       ��K	Q�?��A�?*

logging/current_cost�L<��s�+       ��K	T!@��A�?*

logging/current_cost�4<a�|�+       ��K	^N@��A�?*

logging/current_cost+�<��+       ��K	
}@��A�?*

logging/current_cost4�<���+       ��K	��@��A�?*

logging/current_cost).<"�f~+       ��K	��@��A�?*

logging/current_costF@<���+       ��K	�A��A�?*

logging/current_cost`@<Lt[)+       ��K	7A��A�@*

logging/current_cost<
᠃+       ��K	beA��A�@*

logging/current_cost<mjE�+       ��K	 �A��A�@*

logging/current_costZ`<�� +       ��K	��A��A�@*

logging/current_cost6<�%�+       ��K	��A��A�@*

logging/current_costo<��y+       ��K	�B��A�@*

logging/current_cost�<���+       ��K	WMB��A�@*

logging/current_cost
P<�_��+       ��K	�B��A�@*

logging/current_cost�7<Ŭ��+       ��K	�B��A�@*

logging/current_cost�
<��+       ��K	��B��A�@*

logging/current_cost�<(��+       ��K	UC��A�@*

logging/current_cost�Q<.,ƽ+       ��K	�3C��A�@*

logging/current_cost�.<��+       ��K	�aC��A�@*

logging/current_cost�<��`{+       ��K	��C��A�@*

logging/current_cost�<څ�+       ��K	��C��A�@*

logging/current_cost�O<��+       ��K	��C��A�@*

logging/current_cost[3<�k�w+       ��K	4D��A�@*

logging/current_cost�	<�pi�+       ��K	�CD��A�@*

logging/current_cost�<���+       ��K	�pD��A�@*

logging/current_costjO<k��y+       ��K	��D��A�@*

logging/current_cost.<�B��+       ��K	\�D��A�@*

logging/current_cost�<��CR+       ��K	�D��A�@*

logging/current_cost�<]�[�+       ��K	,E��A�@*

logging/current_cost�B<��Y+       ��K	D]E��A�@*

logging/current_costj<�?L +       ��K	j�E��A�@*

logging/current_cost2�<ζSq+       ��K	��E��A�A*

logging/current_cost�'<"�
�+       ��K	��E��A�A*

logging/current_cost�=<6�\u+       ��K	�F��A�A*

logging/current_cost�,<���X+       ��K	bGF��A�A*

logging/current_cost<"<0��+       ��K	�uF��A�A*

logging/current_cost<�ð:+       ��K	�F��A�A*

logging/current_costg�<8w��+       ��K	H�F��A�A*

logging/current_costTD<E�+       ��K	G��A�A*

logging/current_cost~G<���+       ��K	<2G��A�A*

logging/current_costd<���+       ��K	PaG��A�A*

logging/current_cost��<]-4�+       ��K	h�G��A�A*

logging/current_costm!<XZ�+       ��K	}�G��A�A*

logging/current_cost�T<K*~�+       ��K	H��A�A*

logging/current_cost�<�y7+       ��K	�UH��A�A*

logging/current_cost�<���I+       ��K	B�H��A�A*

logging/current_cost�+<�,��+       ��K	O�H��A�A*

logging/current_costM<r}<�+       ��K	�I��A�A*

logging/current_costd<$1�/+       ��K	GI��A�A*

logging/current_cost��<���+       ��K	��I��A�A*

logging/current_cost�'<@��u+       ��K	�I��A�A*

logging/current_costVP<�M��+       ��K	J��A�A*

logging/current_cost�<�#�+       ��K	aPJ��A�A*

logging/current_cost��<j�"�+       ��K	�J��A�A*

logging/current_cost�.<����+       ��K	�J��A�A*

logging/current_cost�*<�C#+       ��K	��J��A�A*

logging/current_cost�<Kn��+       ��K	�:K��A�A*

logging/current_cost�<#믘+       ��K	�kK��A�B*

logging/current_cost_#<����+       ��K	��K��A�B*

logging/current_cost�<7�aS+       ��K	��K��A�B*

logging/current_cost�0<�+       ��K	K�K��A�B*

logging/current_costC!<��ӄ+       ��K	5,L��A�B*

logging/current_cost{$<��r@+       ��K	1`L��A�B*

logging/current_costS<�;+       ��K	x�L��A�B*

logging/current_cost<�B�+       ��K	`�L��A�B*

logging/current_cost�)<0�İ+       ��K	+�L��A�B*

logging/current_costO-<��+       ��K	M��A�B*

logging/current_cost|<�<�O+       ��K	�IM��A�B*

logging/current_costT9<�6��+       ��K	�yM��A�B*

logging/current_cost�<���|+       ��K	M�M��A�B*

logging/current_cost��<b�Rp+       ��K	s�M��A�B*

logging/current_cost/$<,�D+       ��K	&N��A�B*

logging/current_cost�+<�Ԏ�+       ��K	P8N��A�B*

logging/current_cost�<<r��+       ��K	rN��A�B*

logging/current_cost�"<�V�+       ��K	�N��A�B*

logging/current_costW<��]|+       ��K	b�N��A�B*

logging/current_cost��<��y+       ��K	�N��A�B*

logging/current_cost�/<�ux�+       ��K	�+O��A�B*

logging/current_costdW<��g�+       ��K	�YO��A�B*

logging/current_costb<��p�+       ��K	 �O��A�B*

logging/current_cost�<�T�+       ��K	��O��A�B*

logging/current_cost�=<,��k+       ��K	��O��A�B*

logging/current_cost�<�$��+       ��K	[P��A�B*

logging/current_cost��<*�.�+       ��K	�=P��A�C*

logging/current_cost�-<�U+       ��K	�qP��A�C*

logging/current_cost�#<wv��+       ��K	��P��A�C*

logging/current_cost[<,�1�+       ��K	/�P��A�C*

logging/current_cost��<����+       ��K	_�P��A�C*

logging/current_cost�/<��-+       ��K	Z,Q��A�C*

logging/current_cost+:<;y�:+       ��K	.\Q��A�C*

logging/current_cost�0<X��+       ��K	��Q��A�C*

logging/current_costc2<9~�+       ��K	i�Q��A�C*

logging/current_cost�<���+       ��K	H�Q��A�C*

logging/current_cost�<%���+       ��K	i$R��A�C*

logging/current_cost�<T���+       ��K	�XR��A�C*

logging/current_cost�8<]��+       ��K	m�R��A�C*

logging/current_cost�9<"?�+       ��K	M�R��A�C*

logging/current_cost�<�N�+       ��K	��R��A�C*

logging/current_cost�-<�#�d+       ��K	�S��A�C*

logging/current_cost�;<o��+       ��K	�BS��A�C*

logging/current_cost�<<���+       ��K	�pS��A�C*

logging/current_cost�<���+       ��K	حS��A�C*

logging/current_cost(<|���+       ��K	�S��A�C*

logging/current_cost�?<���&+       ��K	 T��A�C*

logging/current_cost1!<U��=+       ��K	<T��A�C*

logging/current_cost��< W �+       ��K	2kT��A�C*

logging/current_costm/<æ�r+       ��K	3�T��A�C*

logging/current_cost�-<s��+       ��K	3�T��A�C*

logging/current_cost�<���+       ��K	?U��A�D*

logging/current_cost<��$+       ��K	9=U��A�D*

logging/current_cost�-<��m+       ��K	�kU��A�D*

logging/current_cost�(<���X+       ��K	ИU��A�D*

logging/current_cost�<�W?_+       ��K	��U��A�D*

logging/current_cost�<�A�+       ��K	��U��A�D*

logging/current_costLR<��Bo+       ��K	f-V��A�D*

logging/current_cost�:<���+       ��K	�[V��A�D*

logging/current_costI#<�틘+       ��K	G�V��A�D*

logging/current_cost�	<D�5+       ��K	9�V��A�D*

logging/current_cost)�<q�+       ��K	*�V��A�D*

logging/current_cost%$<��)+       ��K	�W��A�D*

logging/current_cost�P<ιL+       ��K	�FW��A�D*

logging/current_cost�<<3�R�+       ��K	�sW��A�D*

logging/current_cost�<��+       ��K	�W��A�D*

logging/current_cost��<0�b�+       ��K	:�W��A�D*

logging/current_cost�<�5��+       ��K	��W��A�D*

logging/current_cost�H<� y+       ��K	�/X��A�D*

logging/current_costx@<�hX+       ��K	)\X��A�D*

logging/current_cost!<svF%+       ��K	g�X��A�D*

logging/current_cost�<ZFd�+       ��K	1�X��A�D*

logging/current_cost��<țǃ+       ��K	��X��A�D*

logging/current_cost�$<�Y�{+       ��K	.Y��A�D*

logging/current_costLI<�=��+       ��K	�JY��A�D*

logging/current_costW;<���+       ��K	�xY��A�D*

logging/current_cost�<	Gg1+       ��K	��Y��A�D*

logging/current_cost��<���l+       ��K	�Y��A�E*

logging/current_cost*<#�Y+       ��K	�Z��A�E*

logging/current_costG</c�;+       ��K	E1Z��A�E*

logging/current_cost�<<�LHF+       ��K	9bZ��A�E*

logging/current_cost� <��
+       ��K	Z�Z��A�E*

logging/current_costT<Q�\�+       ��K	�Z��A�E*

logging/current_cost_�<�0+       ��K	��Z��A�E*

logging/current_cost0%<���+       ��K	�[��A�E*

logging/current_costG<)-��+       ��K	�F[��A�E*

logging/current_costg9<���p+       ��K	�t[��A�E*

logging/current_cost�	<Ҷd+       ��K	�[��A�E*

logging/current_cost��<r�}+       ��K	��[��A�E*

logging/current_cost�<���+       ��K	U�[��A�E*

logging/current_costF<�7�+       ��K	m,\��A�E*

logging/current_cost;<$�d�+       ��K	-Y\��A�E*

logging/current_costq <��Y�+       ��K	/�\��A�E*

logging/current_costK	<
�+       ��K	�\��A�E*

logging/current_cost=�<�mH+       ��K	��\��A�E*

logging/current_cost!%<�P�+       ��K	�]��A�E*

logging/current_cost�E<I+       ��K	=]��A�E*

logging/current_costk8<qb�+       ��K	�j]��A�E*

logging/current_cost<
<���+       ��K	�]��A�E*

logging/current_cost��<���+       ��K	��]��A�E*

logging/current_cost�<4s�G+       ��K	7�]��A�E*

logging/current_costdE<�[g+       ��K	'^��A�E*

logging/current_cost*:<�Jv+       ��K	�U^��A�F*

logging/current_cost5 <��+       ��K	��^��A�F*

logging/current_cost�	<
��	+       ��K	��^��A�F*

logging/current_cost��<mR\+       ��K	2�^��A�F*

logging/current_cost%<�3�+       ��K	-_��A�F*

logging/current_costE<�ҧ.+       ��K	E_��A�F*

logging/current_cost�7<��n`+       ��K	$s_��A�F*

logging/current_costl
<��>w+       ��K	S�_��A�F*

logging/current_cost	�<����+       ��K	��_��A�F*

logging/current_cost�<�5�+       ��K	��_��A�F*

logging/current_cost�D<�Sѕ+       ��K	�+`��A�F*

logging/current_cost�9<K0D�+       ��K	FY`��A�F*

logging/current_cost <ʹ��+       ��K	k�`��A�F*

logging/current_cost�	<��m}+       ��K	v�`��A�F*

logging/current_cost��<F0"+       ��K	��`��A�F*

logging/current_cost�$<�+5�+       ��K	�a��A�F*

logging/current_cost�D<p�,+       ��K	�Ga��A�F*

logging/current_cost�7<�W�[+       ��K	wa��A�F*

logging/current_cost
<����+       ��K	��a��A�F*

logging/current_costD�<�=�r+       ��K	��a��A�F*

logging/current_cost�<��c�+       ��K	�b��A�F*

logging/current_cost�D<�5�b+       ��K	�3b��A�F*

logging/current_costN9<���w+       ��K	(bb��A�F*

logging/current_cost�<���+       ��K	s�b��A�F*

logging/current_cost�	<�)�V+       ��K	��b��A�F*

logging/current_cost��<�8�k+       ��K	P�b��A�F*

logging/current_cost�$<{��A+       ��K	s!c��A�G*

logging/current_costID<$�>�+       ��K	�Mc��A�G*

logging/current_costE7<����+       ��K	�c��A�G*

logging/current_cost�
<�a�+       ��K	m�c��A�G*

logging/current_costi�<�ə�+       ��K	��c��A�G*

logging/current_cost�<=( +       ��K	O	d��A�G*

logging/current_costaD< .K+       ��K	�7d��A�G*

logging/current_cost9<��L�+       ��K	#ed��A�G*

logging/current_cost�<�O�+       ��K	S�d��A�G*

logging/current_cost�	<��cR+       ��K	{�d��A�G*

logging/current_cost�<���+       ��K	��d��A�G*

logging/current_cost�$<rk��+       ��K	�!e��A�G*

logging/current_costD<��%+       ��K	�Qe��A�G*

logging/current_cost7<�r�+       ��K	�~e��A�G*

logging/current_cost�
<UJ�+       ��K	�e��A�G*

logging/current_costp�<߶_.+       ��K	��e��A�G*

logging/current_cost�<�~��+       ��K	�f��A�G*

logging/current_cost5D<�?�+       ��K	64f��A�G*

logging/current_cost�8<U&�v+       ��K	bf��A�G*

logging/current_cost�<�V��+       ��K	^�f��A�G*

logging/current_cost�	<��ٷ+       ��K	~�f��A�G*

logging/current_cost��<<�En+       ��K	��f��A�G*

logging/current_cost�$<}52�+       ��K	�-g��A�G*

logging/current_cost�C<���+       ��K	^g��A�G*

logging/current_cost7<~��0+       ��K	��g��A�G*

logging/current_cost|
<on�*+       ��K	��g��A�G*

logging/current_costn�<MRz�+       ��K	��g��A�H*

logging/current_cost�<�E�+       ��K	�h��A�H*

logging/current_costD<��Pe+       ��K	�Eh��A�H*

logging/current_cost�8<˅��+       ��K	�sh��A�H*

logging/current_cost�<��9�+       ��K	4�h��A�H*

logging/current_cost�	<�P+       ��K	B�h��A�H*

logging/current_cost��<S�+       ��K	��h��A�H*

logging/current_cost�$<�<G�+       ��K	*i��A�H*

logging/current_cost�C<?�^�+       ��K	�Xi��A�H*

logging/current_cost�6<��@3+       ��K	��i��A�H*

logging/current_costk
<i_n+       ��K	��i��A�H*

logging/current_costm�<��i�+       ��K	��i��A�H*

logging/current_cost�<�H~;+       ��K	"j��A�H*

logging/current_cost�C<�K��+       ��K	�8j��A�H*

logging/current_cost�8<�n8B+       ��K	�gj��A�H*

logging/current_costm<]�+       ��K	!�j��A�H*

logging/current_cost�	<+�R+       ��K	)�j��A�H*

logging/current_cost��<g��+       ��K	��j��A�H*

logging/current_cost�$<'��+       ��K	�k��A�H*

logging/current_cost�C<(��;+       ��K	�Kk��A�H*

logging/current_cost�6<�w0l+       ��K	}zk��A�H*

logging/current_cost[
<�e�+       ��K	.�k��A�H*

logging/current_cost\�<�/#�+       ��K	��k��A�H*

logging/current_cost�<���?+       ��K	A
l��A�H*

logging/current_cost�C<��j+       ��K	::l��A�H*

logging/current_cost�8<�+       ��K	`gl��A�I*

logging/current_cost\<�[+       ��K	��l��A�I*

logging/current_cost�	<���`+       ��K	��l��A�I*

logging/current_cost��<���+       ��K	p�l��A�I*

logging/current_costk$<Z��,+       ��K	�m��A�I*

logging/current_cost�C<E��,+       ��K	�Hm��A�I*

logging/current_cost�6<�3X�+       ��K	�wm��A�I*

logging/current_cost@
<���g+       ��K	��m��A�I*

logging/current_costT�<E��+       ��K	��m��A�I*

logging/current_cost�<�GO+       ��K	Bn��A�I*

logging/current_cost�C<�d�+       ��K	W4n��A�I*

logging/current_cost�8<?��+       ��K	�cn��A�I*

logging/current_costD<�c�+       ��K	�n��A�I*

logging/current_cost�	<_~d�+       ��K	\�n��A�I*

logging/current_cost��<�< +       ��K	��n��A�I*

logging/current_costc$<��.�+       ��K	�o��A�I*

logging/current_cost{C<���+       ��K	5Lo��A�I*

logging/current_cost�6<sԳ+       ��K	vyo��A�I*

logging/current_cost8
<��d[+       ��K	��o��A�I*

logging/current_costE�<ݪ�S+       ��K	-�o��A�I*

logging/current_costr<;)��+       ��K		p��A�I*

logging/current_cost�C<0���+       ��K	�2p��A�I*

logging/current_cost�8<��+       ��K	�bp��A�I*

logging/current_cost4<i�^�+       ��K	e�p��A�I*

logging/current_cost�	<go�+       ��K	�p��A�I*

logging/current_costp�<�\�/+       ��K	��p��A�I*

logging/current_costS$<Z��+       ��K	�q��A�J*

logging/current_costeC<E-b+       ��K	�Kq��A�J*

logging/current_cost�6<�E�+       ��K	zyq��A�J*

logging/current_cost!
<lUv�+       ��K	�q��A�J*

logging/current_cost<�<���+       ��K	��q��A�J*

logging/current_cost[<W�(O+       ��K	r��A�J*

logging/current_cost�C<hت�+       ��K	�7r��A�J*

logging/current_cost�8<���+       ��K	�er��A�J*

logging/current_cost"<�y��+       ��K	.�r��A�J*

logging/current_cost]	<GCUU+       ��K	��r��A�J*

logging/current_cost[�<o��+       ��K	��r��A�J*

logging/current_costE$<�T0�+       ��K	bs��A�J*

logging/current_costSC<[�+       ��K	mKs��A�J*

logging/current_cost�6<*��+       ��K	mxs��A�J*

logging/current_cost
<��3J+       ��K	h�s��A�J*

logging/current_cost0�<�pg&+       ��K	��s��A�J*

logging/current_costB<�:X+       ��K	_ t��A�J*

logging/current_cost|C<���r+       ��K	�.t��A�J*

logging/current_cost�8<�W�|+       ��K	�\t��A�J*

logging/current_cost<��7�+       ��K	F�t��A�J*

logging/current_costC	<_��+       ��K	��t��A�J*

logging/current_cost<�<�Ь+       ��K	F�t��A�J*

logging/current_cost1$<�[�l+       ��K	�u��A�J*

logging/current_cost@C<f-�+       ��K	�Cu��A�J*

logging/current_cost�6<Ȯ?+       ��K	�ru��A�J*

logging/current_cost�	<�Xe_+       ��K	��u��A�K*

logging/current_cost$�<���?+       ��K	}�u��A�K*

logging/current_cost,<��f�+       ��K	R�u��A�K*

logging/current_costuC<�[ۼ+       ��K	�,v��A�K*

logging/current_cost�7<P}qV+       ��K	2Zv��A�K*

logging/current_costy<ܵ$p+       ��K	�v��A�K*

logging/current_cost�	<n$<+       ��K	��v��A�K*

logging/current_cost��<Q��t+       ��K	��v��A�K*

logging/current_costd%<�[�+       ��K	w��A�K*

logging/current_cost�E<ʡ�+       ��K	�@w��A�K*

logging/current_cost5<9�΍+       ��K	�mw��A�K*

logging/current_costg
<����+       ��K	s�w��A�K*

logging/current_cost��<`��+       ��K	��w��A�K*

logging/current_cost�<0F�|+       ��K	��w��A�K*

logging/current_cost�A<2e�+       ��K	; x��A�K*

logging/current_cost�6<=iVU+       ��K	2Mx��A�K*

logging/current_costy<���+       ��K	(}x��A�K*

logging/current_cost�<|��+       ��K	�x��A�K*

logging/current_cost��<HH+       ��K	Y�x��A�K*

logging/current_cost`(<ė}�+       ��K	�y��A�K*

logging/current_costSD<38+       ��K	7;y��A�K*

logging/current_cost�5<��o�+       ��K	�ky��A�K*

logging/current_costF<O�v+       ��K	��y��A�K*

logging/current_cost��<�$R�+       ��K	��y��A�K*

logging/current_cost*(<:h)+       ��K	��y��A�K*

logging/current_cost�M<�-�Z+       ��K	!z��A�K*

logging/current_costu/<���+       ��K	�Nz��A�L*

logging/current_cost<r��+       ��K	�{z��A�L*

logging/current_cost<NĈ9+       ��K	�z��A�L*

logging/current_costV<L^�+       ��K	r�z��A�L*

logging/current_cost{;<�"�;+       ��K	�{��A�L*

logging/current_costt <��X+       ��K	3?{��A�L*

logging/current_cost<�#�+       ��K	��{��A�L*

logging/current_cost�U<���B+       ��K	%�{��A�L*

logging/current_costd/<��+       ��K	�&|��A�L*

logging/current_cost�<��+       ��K	Lf|��A�L*

logging/current_cost�<�_�+       ��K	�|��A�L*

logging/current_costR<}}k�+       ��K	��|��A�L*

logging/current_costl1<��c+       ��K		}��A�L*

logging/current_cost�<q��+       ��K	IV}��A�L*

logging/current_cost2<�V��+       ��K	��}��A�L*

logging/current_cost�R<��1�+       ��K	~�}��A�L*

logging/current_cost�0<;��n+       ��K	d�}��A�L*

logging/current_cost/<�!�+       ��K	�3~��A�L*

logging/current_cost{<�E�?+       ��K	to~��A�L*

logging/current_costQ<�� �+       ��K	:�~��A�L*

logging/current_cost�3<���+       ��K	�~��A�L*

logging/current_cost�	<J�h_+       ��K	���A�L*

logging/current_cost�<�?M+       ��K	34��A�L*

logging/current_costN<畿�+       ��K	�h��A�L*

logging/current_cost	3<��2
+       ��K	7���A�L*

logging/current_cost�
<q�+       ��K	d���A�L*

logging/current_costR<9RCh+       ��K	����A�M*

logging/current_costEM<d��+       ��K	����A�M*

logging/current_cost�0<3�k�+       ��K	oM���A�M*

logging/current_costJ
<!�}B+       ��K	X~���A�M*

logging/current_cost?<Bp�+       ��K	ԫ���A�M*

logging/current_cost�M<3�9
+       ��K	�ۀ��A�M*

logging/current_cost�,<���l+       ��K		���A�M*

logging/current_cost�<�eg�+       ��K	7���A�M*

logging/current_costu<���<+       ��K	�{���A�M*

logging/current_cost�@<o�0+       ��K	|����A�M*

logging/current_costu<�g�*+       ��K	�끲�A�M*

logging/current_cost�,<�(�<+       ��K	&���A�M*

logging/current_cost�+<E/�+       ��K	�f���A�M*

logging/current_costp(<M�(+       ��K	:����A�M*

logging/current_costi<����+       ��K	>܂��A�M*

logging/current_cost{+<��+       ��K	����A�M*

logging/current_costy<>���+       ��K	+E���A�M*

logging/current_cost<��d+       ��K	�����A�M*

logging/current_cost�<ܣ��+       ��K	A����A�M*

logging/current_cost<+�M+       ��K	�惲�A�M*

logging/current_cost-<�6+       ��K	����A�M*

logging/current_cost@$<���/+       ��K	*R���A�M*

logging/current_cost<<��U,+       ��K	�����A�M*

logging/current_cost�#<Ul+       ��K	-����A�M*

logging/current_cost�<~��+       ��K	鄲�A�M*

logging/current_cost<�W9+       ��K	J���A�N*

logging/current_cost	<,��+       ��K	�F���A�N*

logging/current_cost�"<8�>�+       ��K	rs���A�N*

logging/current_cost��<C^v�