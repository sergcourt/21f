       �K"	   ��Abrain.Event:2х;M�      ��	���A"��
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
/layer_2/weights2/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *#
_class
loc:@layer_2/weights2*
valueB
 *׳]�
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
+layer_3/weights3/Initializer/random_uniformAdd/layer_3/weights3/Initializer/random_uniform/mul/layer_3/weights3/Initializer/random_uniform/min*
_output_shapes

:*
T0*#
_class
loc:@layer_3/weights3
�
layer_3/weights3
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
layer_3/biases3/AssignAssignlayer_3/biases3!layer_3/biases3/Initializer/zeros*
T0*"
_class
loc:@layer_3/biases3*
validate_shape(*
_output_shapes
:*
use_locking(
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
VariableV2*
shared_name *"
_class
loc:@output/weights4*
	container *
shape
:*
dtype0*
_output_shapes

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
output/weights4/readIdentityoutput/weights4*
_output_shapes

:*
T0*"
_class
loc:@output/weights4
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
output/biases4/AssignAssignoutput/biases4 output/biases4/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*!
_class
loc:@output/biases4
w
output/biases4/readIdentityoutput/biases4*
_output_shapes
:*
T0*!
_class
loc:@output/biases4
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
#train/gradients/cost/Mean_grad/TileTile&train/gradients/cost/Mean_grad/Reshape$train/gradients/cost/Mean_grad/Shape*
T0*'
_output_shapes
:���������*

Tmultiples0
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
$train/gradients/cost/Mean_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
�
#train/gradients/cost/Mean_grad/ProdProd&train/gradients/cost/Mean_grad/Shape_1$train/gradients/cost/Mean_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
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
3train/gradients/cost/SquaredDifference_grad/Shape_1Shapecost/Placeholder*
_output_shapes
:*
T0*
out_type0
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
output/addcost/Placeholder'^train/gradients/cost/Mean_grad/truediv*
T0*'
_output_shapes
:���������
�
1train/gradients/cost/SquaredDifference_grad/mul_1Mul/train/gradients/cost/SquaredDifference_grad/mul/train/gradients/cost/SquaredDifference_grad/sub*'
_output_shapes
:���������*
T0
�
/train/gradients/cost/SquaredDifference_grad/SumSum1train/gradients/cost/SquaredDifference_grad/mul_1Atrain/gradients/cost/SquaredDifference_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
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
%train/gradients/output/add_grad/ShapeShapeoutput/MatMul*
_output_shapes
:*
T0*
out_type0
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
*train/gradients/layer_3/Relu_grad/ReluGradReluGrad;train/gradients/output/MatMul_grad/tuple/control_dependencylayer_3/Relu*'
_output_shapes
:���������*
T0
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
*train/gradients/layer_3/MatMul_grad/MatMulMatMul9train/gradients/layer_3/add_grad/tuple/control_dependencylayer_3/weights3/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b(
�
,train/gradients/layer_3/MatMul_grad/MatMul_1MatMullayer_2/Relu9train/gradients/layer_3/add_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:*
transpose_a(
�
4train/gradients/layer_3/MatMul_grad/tuple/group_depsNoOp+^train/gradients/layer_3/MatMul_grad/MatMul-^train/gradients/layer_3/MatMul_grad/MatMul_1
�
<train/gradients/layer_3/MatMul_grad/tuple/control_dependencyIdentity*train/gradients/layer_3/MatMul_grad/MatMul5^train/gradients/layer_3/MatMul_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*=
_class3
1/loc:@train/gradients/layer_3/MatMul_grad/MatMul
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
(train/gradients/layer_2/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
6train/gradients/layer_2/add_grad/BroadcastGradientArgsBroadcastGradientArgs&train/gradients/layer_2/add_grad/Shape(train/gradients/layer_2/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
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
&train/gradients/layer_2/add_grad/Sum_1Sum*train/gradients/layer_2/Relu_grad/ReluGrad8train/gradients/layer_2/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
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
<train/gradients/layer_2/MatMul_grad/tuple/control_dependencyIdentity*train/gradients/layer_2/MatMul_grad/MatMul5^train/gradients/layer_2/MatMul_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*=
_class3
1/loc:@train/gradients/layer_2/MatMul_grad/MatMul
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
9train/gradients/layer_1/add_grad/tuple/control_dependencyIdentity(train/gradients/layer_1/add_grad/Reshape2^train/gradients/layer_1/add_grad/tuple/group_deps*
T0*;
_class1
/-loc:@train/gradients/layer_1/add_grad/Reshape*'
_output_shapes
:���������
�
;train/gradients/layer_1/add_grad/tuple/control_dependency_1Identity*train/gradients/layer_1/add_grad/Reshape_12^train/gradients/layer_1/add_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_1/add_grad/Reshape_1*
_output_shapes
:
�
*train/gradients/layer_1/MatMul_grad/MatMulMatMul9train/gradients/layer_1/add_grad/tuple/control_dependencylayer_1/weights1/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b(
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
>train/gradients/layer_1/MatMul_grad/tuple/control_dependency_1Identity,train/gradients/layer_1/MatMul_grad/MatMul_15^train/gradients/layer_1/MatMul_grad/tuple/group_deps*
_output_shapes

:*
T0*?
_class5
31loc:@train/gradients/layer_1/MatMul_grad/MatMul_1
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
train/beta1_power/AssignAssigntrain/beta1_powertrain/beta1_power/initial_value*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
: *
use_locking(
z
train/beta1_power/readIdentitytrain/beta1_power*
T0*"
_class
loc:@layer_1/biases1*
_output_shapes
: 
�
train/beta2_power/initial_valueConst*
dtype0*
_output_shapes
: *"
_class
loc:@layer_1/biases1*
valueB
 *w�?
�
train/beta2_power
VariableV2*"
_class
loc:@layer_1/biases1*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
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
/train/layer_1/weights1/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes

:*#
_class
loc:@layer_1/weights1*
valueB*    
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
$train/layer_1/weights1/Adam_1/AssignAssigntrain/layer_1/weights1/Adam_1/train/layer_1/weights1/Adam_1/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@layer_1/weights1*
validate_shape(*
_output_shapes

:
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
-train/layer_2/weights2/Adam/Initializer/zerosConst*
dtype0*
_output_shapes

:*#
_class
loc:@layer_2/weights2*
valueB*    
�
train/layer_2/weights2/Adam
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
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@layer_2/biases2
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
!train/layer_2/biases2/Adam_1/readIdentitytrain/layer_2/biases2/Adam_1*
_output_shapes
:*
T0*"
_class
loc:@layer_2/biases2
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
"train/layer_3/weights3/Adam/AssignAssigntrain/layer_3/weights3/Adam-train/layer_3/weights3/Adam/Initializer/zeros*
T0*#
_class
loc:@layer_3/weights3*
validate_shape(*
_output_shapes

:*
use_locking(
�
 train/layer_3/weights3/Adam/readIdentitytrain/layer_3/weights3/Adam*
_output_shapes

:*
T0*#
_class
loc:@layer_3/weights3
�
/train/layer_3/weights3/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes

:*#
_class
loc:@layer_3/weights3*
valueB*    
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
VariableV2*
shared_name *"
_class
loc:@output/weights4*
	container *
shape
:*
dtype0*
_output_shapes

:
�
!train/output/weights4/Adam/AssignAssigntrain/output/weights4/Adam,train/output/weights4/Adam/Initializer/zeros*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*"
_class
loc:@output/weights4
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
 train/output/biases4/Adam/AssignAssigntrain/output/biases4/Adam+train/output/biases4/Adam/Initializer/zeros*
T0*!
_class
loc:@output/biases4*
validate_shape(*
_output_shapes
:*
use_locking(
�
train/output/biases4/Adam/readIdentitytrain/output/biases4/Adam*
_output_shapes
:*
T0*!
_class
loc:@output/biases4
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
train/Adam/mulMultrain/beta1_power/readtrain/Adam/beta1,^train/Adam/update_layer_1/biases1/ApplyAdam-^train/Adam/update_layer_1/weights1/ApplyAdam,^train/Adam/update_layer_2/biases2/ApplyAdam-^train/Adam/update_layer_2/weights2/ApplyAdam,^train/Adam/update_layer_3/biases3/ApplyAdam-^train/Adam/update_layer_3/weights3/ApplyAdam+^train/Adam/update_output/biases4/ApplyAdam,^train/Adam/update_output/weights4/ApplyAdam*
_output_shapes
: *
T0*"
_class
loc:@layer_1/biases1
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
save/AssignAssignlayer_1/biases1save/RestoreV2*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@layer_1/biases1
�
save/Assign_1Assignlayer_1/weights1save/RestoreV2:1*
T0*#
_class
loc:@layer_1/weights1*
validate_shape(*
_output_shapes

:*
use_locking(
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
save/Assign_4Assignlayer_3/biases3save/RestoreV2:4*
T0*"
_class
loc:@layer_3/biases3*
validate_shape(*
_output_shapes
:*
use_locking(
�
save/Assign_5Assignlayer_3/weights3save/RestoreV2:5*
T0*#
_class
loc:@layer_3/weights3*
validate_shape(*
_output_shapes

:*
use_locking(
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
save/Assign_12Assigntrain/layer_1/weights1/Adamsave/RestoreV2:12*
T0*#
_class
loc:@layer_1/weights1*
validate_shape(*
_output_shapes

:*
use_locking(
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
save/Assign_14Assigntrain/layer_2/biases2/Adamsave/RestoreV2:14*
use_locking(*
T0*"
_class
loc:@layer_2/biases2*
validate_shape(*
_output_shapes
:
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
save/Assign_16Assigntrain/layer_2/weights2/Adamsave/RestoreV2:16*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@layer_2/weights2
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
save/Assign_18Assigntrain/layer_3/biases3/Adamsave/RestoreV2:18*
T0*"
_class
loc:@layer_3/biases3*
validate_shape(*
_output_shapes
:*
use_locking(
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
save/Assign_21Assigntrain/layer_3/weights3/Adam_1save/RestoreV2:21*
use_locking(*
T0*#
_class
loc:@layer_3/weights3*
validate_shape(*
_output_shapes

:
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
initNoOp^layer_1/biases1/Assign^layer_1/weights1/Assign^layer_2/biases2/Assign^layer_2/weights2/Assign^layer_3/biases3/Assign^layer_3/weights3/Assign^output/biases4/Assign^output/weights4/Assign^train/beta1_power/Assign^train/beta2_power/Assign"^train/layer_1/biases1/Adam/Assign$^train/layer_1/biases1/Adam_1/Assign#^train/layer_1/weights1/Adam/Assign%^train/layer_1/weights1/Adam_1/Assign"^train/layer_2/biases2/Adam/Assign$^train/layer_2/biases2/Adam_1/Assign#^train/layer_2/weights2/Adam/Assign%^train/layer_2/weights2/Adam_1/Assign"^train/layer_3/biases3/Adam/Assign$^train/layer_3/biases3/Adam_1/Assign#^train/layer_3/weights3/Adam/Assign%^train/layer_3/weights3/Adam_1/Assign!^train/output/biases4/Adam/Assign#^train/output/biases4/Adam_1/Assign"^train/output/weights4/Adam/Assign$^train/output/weights4/Adam_1/Assign"2[;��     ��d]	 7��AJ܉
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
layer_1/biases1/AssignAssignlayer_1/biases1!layer_1/biases1/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
:
z
layer_1/biases1/readIdentitylayer_1/biases1*
_output_shapes
:*
T0*"
_class
loc:@layer_1/biases1
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
/layer_2/weights2/Initializer/random_uniform/mulMul9layer_2/weights2/Initializer/random_uniform/RandomUniform/layer_2/weights2/Initializer/random_uniform/sub*
T0*#
_class
loc:@layer_2/weights2*
_output_shapes

:
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
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@layer_2/biases2
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
layer_2/MatMulMatMullayer_1/Relulayer_2/weights2/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
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
/layer_3/weights3/Initializer/random_uniform/subSub/layer_3/weights3/Initializer/random_uniform/max/layer_3/weights3/Initializer/random_uniform/min*
_output_shapes
: *
T0*#
_class
loc:@layer_3/weights3
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
layer_3/biases3/AssignAssignlayer_3/biases3!layer_3/biases3/Initializer/zeros*
T0*"
_class
loc:@layer_3/biases3*
validate_shape(*
_output_shapes
:*
use_locking(
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
layer_3/addAddlayer_3/MatMullayer_3/biases3/read*'
_output_shapes
:���������*
T0
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
output/weights4/AssignAssignoutput/weights4*output/weights4/Initializer/random_uniform*
use_locking(*
T0*"
_class
loc:@output/weights4*
validate_shape(*
_output_shapes

:
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
output/biases4/AssignAssignoutput/biases4 output/biases4/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*!
_class
loc:@output/biases4
w
output/biases4/readIdentityoutput/biases4*
T0*!
_class
loc:@output/biases4*
_output_shapes
:
�
output/MatMulMatMullayer_3/Reluoutput/weights4/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
g

output/addAddoutput/MatMuloutput/biases4/read*'
_output_shapes
:���������*
T0
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
cost/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
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
&train/gradients/cost/Mean_grad/truedivRealDiv#train/gradients/cost/Mean_grad/Tile#train/gradients/cost/Mean_grad/Cast*'
_output_shapes
:���������*
T0
{
1train/gradients/cost/SquaredDifference_grad/ShapeShape
output/add*
_output_shapes
:*
T0*
out_type0
�
3train/gradients/cost/SquaredDifference_grad/Shape_1Shapecost/Placeholder*
_output_shapes
:*
T0*
out_type0
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
'train/gradients/output/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
5train/gradients/output/add_grad/BroadcastGradientArgsBroadcastGradientArgs%train/gradients/output/add_grad/Shape'train/gradients/output/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
#train/gradients/output/add_grad/SumSumDtrain/gradients/cost/SquaredDifference_grad/tuple/control_dependency5train/gradients/output/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
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
8train/gradients/output/add_grad/tuple/control_dependencyIdentity'train/gradients/output/add_grad/Reshape1^train/gradients/output/add_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*:
_class0
.,loc:@train/gradients/output/add_grad/Reshape
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
+train/gradients/output/MatMul_grad/MatMul_1MatMullayer_3/Relu8train/gradients/output/add_grad/tuple/control_dependency*
_output_shapes

:*
transpose_a(*
transpose_b( *
T0
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
=train/gradients/output/MatMul_grad/tuple/control_dependency_1Identity+train/gradients/output/MatMul_grad/MatMul_14^train/gradients/output/MatMul_grad/tuple/group_deps*
_output_shapes

:*
T0*>
_class4
20loc:@train/gradients/output/MatMul_grad/MatMul_1
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
<train/gradients/layer_3/MatMul_grad/tuple/control_dependencyIdentity*train/gradients/layer_3/MatMul_grad/MatMul5^train/gradients/layer_3/MatMul_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*=
_class3
1/loc:@train/gradients/layer_3/MatMul_grad/MatMul
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
$train/gradients/layer_1/add_grad/SumSum*train/gradients/layer_1/Relu_grad/ReluGrad6train/gradients/layer_1/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
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
>train/gradients/layer_1/MatMul_grad/tuple/control_dependency_1Identity,train/gradients/layer_1/MatMul_grad/MatMul_15^train/gradients/layer_1/MatMul_grad/tuple/group_deps*
_output_shapes

:*
T0*?
_class5
31loc:@train/gradients/layer_1/MatMul_grad/MatMul_1
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
shape: *
dtype0*
_output_shapes
: *
shared_name *"
_class
loc:@layer_1/biases1*
	container 
�
train/beta1_power/AssignAssigntrain/beta1_powertrain/beta1_power/initial_value*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
: *
use_locking(
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
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *"
_class
loc:@layer_1/biases1
�
train/beta2_power/AssignAssigntrain/beta2_powertrain/beta2_power/initial_value*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
: *
use_locking(
z
train/beta2_power/readIdentitytrain/beta2_power*
T0*"
_class
loc:@layer_1/biases1*
_output_shapes
: 
�
-train/layer_1/weights1/Adam/Initializer/zerosConst*
dtype0*
_output_shapes

:*#
_class
loc:@layer_1/weights1*
valueB*    
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
 train/layer_1/weights1/Adam/readIdentitytrain/layer_1/weights1/Adam*
_output_shapes

:*
T0*#
_class
loc:@layer_1/weights1
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
$train/layer_1/weights1/Adam_1/AssignAssigntrain/layer_1/weights1/Adam_1/train/layer_1/weights1/Adam_1/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@layer_1/weights1*
validate_shape(*
_output_shapes

:
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
!train/layer_1/biases1/Adam/AssignAssigntrain/layer_1/biases1/Adam,train/layer_1/biases1/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@layer_1/biases1
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
.train/layer_2/biases2/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:*"
_class
loc:@layer_2/biases2*
valueB*    
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
-train/layer_3/weights3/Adam/Initializer/zerosConst*#
_class
loc:@layer_3/weights3*
valueB*    *
dtype0*
_output_shapes

:
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
/train/layer_3/weights3/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes

:*#
_class
loc:@layer_3/weights3*
valueB*    
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
$train/layer_3/weights3/Adam_1/AssignAssigntrain/layer_3/weights3/Adam_1/train/layer_3/weights3/Adam_1/Initializer/zeros*
T0*#
_class
loc:@layer_3/weights3*
validate_shape(*
_output_shapes

:*
use_locking(
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
,train/output/weights4/Adam/Initializer/zerosConst*
dtype0*
_output_shapes

:*"
_class
loc:@output/weights4*
valueB*    
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
VariableV2*
shared_name *"
_class
loc:@output/weights4*
	container *
shape
:*
dtype0*
_output_shapes

:
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
 train/output/biases4/Adam/AssignAssigntrain/output/biases4/Adam+train/output/biases4/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*!
_class
loc:@output/biases4
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
train/Adam/epsilonConst*
valueB
 *w�+2*
dtype0*
_output_shapes
: 
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
,train/Adam/update_layer_2/weights2/ApplyAdam	ApplyAdamlayer_2/weights2train/layer_2/weights2/Adamtrain/layer_2/weights2/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon>train/gradients/layer_2/MatMul_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes

:*
use_locking( *
T0*#
_class
loc:@layer_2/weights2
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
+train/Adam/update_layer_3/biases3/ApplyAdam	ApplyAdamlayer_3/biases3train/layer_3/biases3/Adamtrain/layer_3/biases3/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon;train/gradients/layer_3/add_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0*"
_class
loc:@layer_3/biases3
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
save/Assign_1Assignlayer_1/weights1save/RestoreV2:1*
use_locking(*
T0*#
_class
loc:@layer_1/weights1*
validate_shape(*
_output_shapes

:
�
save/Assign_2Assignlayer_2/biases2save/RestoreV2:2*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@layer_2/biases2
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
save/Assign_4Assignlayer_3/biases3save/RestoreV2:4*
T0*"
_class
loc:@layer_3/biases3*
validate_shape(*
_output_shapes
:*
use_locking(
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
save/Assign_9Assigntrain/beta2_powersave/RestoreV2:9*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
: *
use_locking(
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
save/Assign_20Assigntrain/layer_3/weights3/Adamsave/RestoreV2:20*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@layer_3/weights3
�
save/Assign_21Assigntrain/layer_3/weights3/Adam_1save/RestoreV2:21*
use_locking(*
T0*#
_class
loc:@layer_3/weights3*
validate_shape(*
_output_shapes

:
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
train/output/biases4/Adam_1:0"train/output/biases4/Adam_1/Assign"train/output/biases4/Adam_1/read:02/train/output/biases4/Adam_1/Initializer/zeros:0����(       �pJ	���A*

logging/current_cost��
=;�<*       ����	h���A*

logging/current_cost�=���*       ����	���A
*

logging/current_cost"b=�<��*       ����		O��A*

logging/current_cost`1 =ͼ�*       ����	���A*

logging/current_cost�y�<�K�*       ����	����A*

logging/current_cost�<�j*       ����	i���A*

logging/current_cost�6�<<�%*       ����	���A#*

logging/current_cost]��<bʕZ*       ����	�U��A(*

logging/current_costr��<�ߩ�*       ����	����A-*

logging/current_cost�z�<���2*       ����	����A2*

logging/current_cost�o�<7Y6>*       ����	����A7*

logging/current_cost���<���/*       ����	���A<*

logging/current_costAw�<�-l*       ����	�F��AA*

logging/current_cost}x�</�|�*       ����	Rt��AF*

logging/current_costh��<��k*       ����	����AK*

logging/current_cost�L�<����*       ����	1���AP*

logging/current_cost��<N*       ����	����AU*

logging/current_cost��<;"�*       ����	.��AZ*

logging/current_cost$�<�2�*       ����	�\��A_*

logging/current_cost�d�<��� *       ����	]���Ad*

logging/current_cost��<���C*       ����	5���Ai*

logging/current_cost,�<p"�*       ����	����An*

logging/current_cost3��<�=��*       ����	���As*

logging/current_cost���<ו(*       ����	 D��Ax*

logging/current_costӃ�<�h,/*       ����	Qs��A}*

logging/current_cost���<2�r!+       ��K	ȟ��A�*

logging/current_cost���<�	Í+       ��K	*���A�*

logging/current_costS��<��|�+       ��K	����A�*

logging/current_costTD�<Q��+       ��K	�+��A�*

logging/current_cost�]�<�q�j+       ��K	�[��A�*

logging/current_cost��<>��e+       ��K	���A�*

logging/current_costۧ�<@,la+       ��K	���A�*

logging/current_costLպ<QT�+       ��K	����A�*

logging/current_cost5��<��+       ��K	=��A�*

logging/current_costf;�<��+       ��K	�?��A�*

logging/current_costsl�<��+       ��K	�l��A�*

logging/current_costU��<��B�+       ��K	���A�*

logging/current_cost���<+$v+       ��K	����A�*

logging/current_cost���<�`y+       ��K	����A�*

logging/current_cost�4�<�?ѭ+       ��K	h+��A�*

logging/current_costv�<�+       ��K	/\��A�*

logging/current_cost�ɡ<��n�+       ��K	����A�*

logging/current_costK�<ċ(�+       ��K	r���A�*

logging/current_costFf�<���'+       ��K	^���A�*

logging/current_cost�ə<����+       ��K	���A�*

logging/current_cost�7�<�$o+       ��K	f>��A�*

logging/current_costb��<��ڳ+       ��K	bl��A�*

logging/current_cost�;�<=�@o+       ��K	����A�*

logging/current_costf؏<�Z�+       ��K	����A�*

logging/current_cost܈�<�3�P+       ��K	q���A�*

logging/current_cost�+�<խ��+       ��K	Q'��A�*

logging/current_costq�<���l+       ��K	U��A�*

logging/current_cost���<�u��+       ��K	����A�*

logging/current_cost}��<�S�+       ��K	J���A�*

logging/current_cost�s�<��,+       ��K	����A�*

logging/current_cost�o�<�ώ9+       ��K	���A�*

logging/current_cost��|<҅��+       ��K	s5��A�*

logging/current_cost�x<��U+       ��K	�a��A�*

logging/current_costA1u<b�+       ��K	ێ��A�*

logging/current_cost�{q< ��+       ��K	���A�*

logging/current_cost|�m<V>D\+       ��K	����A�*

logging/current_cost�Vj<�@��+       ��K	d��A�*

logging/current_cost]�f<��Do+       ��K	*C��A�*

logging/current_cost�c<���+       ��K	�q��A�*

logging/current_costly`<�B+       ��K	���A�*

logging/current_cost%l]<��+       ��K	����A�*

logging/current_cost�vZ<�i+       ��K	R���A�*

logging/current_costw�W<�O��+       ��K	�'��A�*

logging/current_cost��T<��*�+       ��K	�V��A�*

logging/current_costYR<�X�5+       ��K	���A�*

logging/current_cost�YO<�Br+       ��K	r���A�*

logging/current_cost�L<�Q-�+       ��K	����A�*

logging/current_cost�,J<�R�+       ��K	��A�*

logging/current_costc�G<"�+       ��K	5@��A�*

logging/current_cost�JE<OQ��+       ��K	Co��A�*

logging/current_cost��B<V���+       ��K	���A�*

logging/current_cost�@<�9+       ��K	����A�*

logging/current_cost&p><+NĀ+       ��K	5���A�*

logging/current_cost(D<<��u"+       ��K	F+��A�*

logging/current_costc":<F�^�+       ��K	�Z��A�*

logging/current_costS
8<g��@+       ��K	����A�*

logging/current_cost�6<�[+       ��K	����A�*

logging/current_cost�4<W@��+       ��K	R���A�*

logging/current_cost222<����+       ��K	:��A�*

logging/current_costnP0<��P�+       ��K	!L��A�*

logging/current_cost�.<��9V+       ��K	"{��A�*

logging/current_cost��,<:�-:+       ��K	���A�*

logging/current_cost��*<�My�+       ��K	���A�*

logging/current_cost*)<b_�+       ��K	���A�*

logging/current_cost�w'<�0�+       ��K	x3��A�*

logging/current_cost�%<���+       ��K	fa��A�*

logging/current_costS&$<��Q�+       ��K	ϓ��A�*

logging/current_costN�"<#'�0+       ��K	����A�*

logging/current_costh� <�ᩃ+       ��K	,���A�*

logging/current_cost�_<g �m+       ��K	�)��A�*

logging/current_cost��<�f�#+       ��K	�X��A�*

logging/current_cost"@<�-p�+       ��K	����A�*

logging/current_cost�<W���+       ��K	����A�*

logging/current_cost<=QE�+       ��K	����A�*

logging/current_cost1�<+?��+       ��K	���A�*

logging/current_cost�<��+       ��K	+A��A�*

logging/current_costi�<�7]+       ��K	 p��A�*

logging/current_cost	<�{-v+       ��K		���A�*

logging/current_costϚ<C��g+       ��K	����A�*

logging/current_cost71<w��N+       ��K	V���A�*

logging/current_cost�<Ԫ5+       ��K	�& ��A�*

logging/current_cost{|<o!Z�+       ��K	ET ��A�*

logging/current_cost�,<�0p�+       ��K	� ��A�*

logging/current_cost��
<��_�+       ��K	P� ��A�*

logging/current_costL�	<��I+       ��K	� ��A�*

logging/current_cost0<l@�$+       ��K	5!��A�*

logging/current_cost1U<m��p+       ��K	;<!��A�*

logging/current_cost�4<8�.+       ��K	�i!��A�*

logging/current_costB<-U��+       ��K	-�!��A�*

logging/current_cost<׾�+       ��K	h�!��A�*

logging/current_cost��<��bL+       ��K	��!��A�*

logging/current_cost��<�u�z+       ��K	"��A�*

logging/current_cost�� <�1�a+       ��K	�L"��A�*

logging/current_cost��;�LS-+       ��K	�y"��A�*

logging/current_cost-��;[��+       ��K	˥"��A�*

logging/current_cost3��;5��+       ��K	��"��A�*

logging/current_cost���;����+       ��K	#��A�*

logging/current_costx��;���m+       ��K	�.#��A�*

logging/current_cost�;�l��+       ��K	:[#��A�*

logging/current_cost7�;�;��+       ��K	��#��A�*

logging/current_cost|g�;�B?*+       ��K	��#��A�*

logging/current_cost���;Ns�+       ��K	W�#��A�*

logging/current_cost��;��'d+       ��K	�$��A�*

logging/current_cost:.�;��W�+       ��K	�=$��A�*

logging/current_cost��;�oɫ+       ��K	wj$��A�*

logging/current_cost���;F���+       ��K	J�$��A�*

logging/current_costT@�;���V+       ��K	��$��A�*

logging/current_cost���;�#�q+       ��K	g�$��A�*

logging/current_cost��; �@�+       ��K	�#%��A�*

logging/current_costF��;)>�+       ��K	Q%��A�*

logging/current_cost���;���+       ��K	�}%��A�*

logging/current_cost�V�;h�J�+       ��K	�%��A�*

logging/current_cost���;��@�+       ��K	X�%��A�*

logging/current_cost8Y�;q��+       ��K	T&��A�*

logging/current_cost���;�yk�+       ��K	�4&��A�*

logging/current_cost$q�;x��E+       ��K	!c&��A�*

logging/current_cost��;��]G+       ��K	N�&��A�*

logging/current_cost���;e��Z+       ��K	��&��A�*

logging/current_cost?6�;���+       ��K	v'��A�*

logging/current_cost���;���+       ��K	�M'��A�*

logging/current_cost[��;|C�+       ��K	Rz'��A�*

logging/current_cost#.�;�ͦ�+       ��K	��'��A�*

logging/current_cost���;���+       ��K	�'��A�*

logging/current_costl��;�N�A+       ��K		(��A�*

logging/current_cost�V�;`H�?+       ��K	�5(��A�*

logging/current_cost��;����+       ��K	�e(��A�*

logging/current_cost��;v6�j+       ��K	P�(��A�*

logging/current_cost"��;�G�+       ��K	��(��A�*

logging/current_cost�u�;����+       ��K	��(��A�*

logging/current_costVH�;ǰXT+       ��K	\#)��A�*

logging/current_cost��;�V+       ��K	@T)��A�*

logging/current_cost���;9�]�+       ��K	�)��A�*

logging/current_cost���;1yC+       ��K	R�)��A�*

logging/current_cost���;�Ê�+       ��K	9�)��A�*

logging/current_cost=��;sqm+       ��K	_*��A�*

logging/current_cost��;v�9+       ��K	:<*��A�*

logging/current_cost���;���4+       ��K	�i*��A�*

logging/current_costq��;���q+       ��K	��*��A�*

logging/current_cost,��;���+       ��K	��*��A�*

logging/current_cost���;��H+       ��K	��*��A�*

logging/current_costߒ�;��+       ��K	�#+��A�*

logging/current_costz��;<�$�+       ��K	�R+��A�*

logging/current_costܪ�;��+       ��K	�+��A�*

logging/current_costp��;�+       ��K	��+��A�*

logging/current_costӸ;ޞ3+       ��K	@�+��A�*

logging/current_cost$�;��+       ��K	�,��A�*

logging/current_cost��;c�F+       ��K	�;,��A�*

logging/current_cost��;�]+       ��K	*i,��A�*

logging/current_cost�5�;�G+       ��K	o�,��A�*

logging/current_costKQ�;�u�+       ��K	��,��A�*

logging/current_cost�o�;�*�+       ��K	v�,��A�*

logging/current_cost���;O�q�+       ��K	� -��A�*

logging/current_cost���;?y@�+       ��K	�M-��A�*

logging/current_costa߰;Y�+       ��K	Fz-��A�*

logging/current_costm�; e{+       ��K	��-��A�*

logging/current_costA�;�`+       ��K	�-��A�*

logging/current_cost�y�;���/+       ��K	�	.��A�*

logging/current_cost\��;D�+       ��K	#8.��A�*

logging/current_cost��;Z���+       ��K	Pe.��A�*

logging/current_cost'-�;v3Z+       ��K	d�.��A�*

logging/current_costMp�;e�T�+       ��K	S�.��A�*

logging/current_costP��;z8G+       ��K	|�.��A�*

logging/current_costI��;�\~�+       ��K	�/��A�*

logging/current_costJ�;��é+       ��K	RI/��A�*

logging/current_costO��;����+       ��K	�w/��A�*

logging/current_cost��;�J#+       ��K	��/��A�*

logging/current_cost0B�;�� 6+       ��K	�/��A�*

logging/current_costS��;w-�+       ��K	D�/��A�*

logging/current_cost���;��Ӏ+       ��K	)0��A�*

logging/current_costQ[�;,��T+       ��K	iU0��A�*

logging/current_cost_��;�]-+       ��K	:�0��A�*

logging/current_costA!�;�B�|+       ��K	E�0��A�*

logging/current_costH��;�D�+       ��K	��0��A�*

logging/current_cost��;��yF+       ��K	�
1��A�*

logging/current_cost�`�;���+       ��K		=1��A�*

logging/current_costjС;'z}+       ��K	�i1��A�*

logging/current_cost?�;B�
t+       ��K	-�1��A�*

logging/current_cost���;ջw+       ��K	�1��A�*

logging/current_costE"�;6�!�+       ��K	�1��A�*

logging/current_cost���;��,+       ��K	�2��A�*

logging/current_costm�;�[+       ��K	�K2��A�*

logging/current_costw��;_���+       ��K	�y2��A�*

logging/current_costH�;�+6�+       ��K	h�2��A�*

logging/current_cost
��;���B+       ��K	�2��A�*

logging/current_cost�; ��*+       ��K	#3��A�*

logging/current_cost���;Ņ��+       ��K	/3��A�*

logging/current_cost��;��@+       ��K	S[3��A�*

logging/current_cost���;uKU+       ��K	{�3��A�*

logging/current_cost�=�;��L�+       ��K	-�3��A�*

logging/current_cost=Ԛ;rX!+       ��K	��3��A�*

logging/current_costig�;̱��+       ��K	�4��A�*

logging/current_cost��;��t=+       ��K	�G4��A�*

logging/current_cost���;����+       ��K	w4��A�*

logging/current_costA�;x���+       ��K	��4��A�*

logging/current_cost"�;4��e+       ��K	G�4��A�*

logging/current_cost���;ʱw�+       ��K	�5��A�*

logging/current_cost�.�;X''+       ��K	�.5��A�*

logging/current_costؗ;�mD�+       ��K	(\5��A�*

logging/current_cost���;��+       ��K	�5��A�*

logging/current_costP3�;�QKz+       ��K	ո5��A�*

logging/current_costt�;�t�+       ��K	B�5��A�*

logging/current_cost���;�L��+       ��K	�6��A�*

logging/current_cost,H�;X���+       ��K	�D6��A�*

logging/current_cost���;� .+       ��K	3u6��A�*

logging/current_cost��;��+       ��K	x�6��A�*

logging/current_cost�o�;�?�+       ��K	a�6��A�	*

logging/current_cost�,�;VI�+       ��K	6 7��A�	*

logging/current_cost�;g�;�+       ��K	Y07��A�	*

logging/current_cost-��;dG�#+       ��K	^\7��A�	*

logging/current_costf�;�Sň+       ��K	ˉ7��A�	*

logging/current_cost&(�;H�
9+       ��K	J�7��A�	*

logging/current_cost�;H��+       ��K	��7��A�	*

logging/current_costӯ�;��+       ��K	8��A�	*

logging/current_cost3w�;����+       ��K	:C8��A�	*

logging/current_cost�>�;m�4+       ��K	t8��A�	*

logging/current_cost"�;��]�+       ��K	�8��A�	*

logging/current_costВ;FfI�+       ��K	��8��A�	*

logging/current_cost���;?a�Q+       ��K	��8��A�	*

logging/current_costCh�;>%=+       ��K	"19��A�	*

logging/current_costv7�;�A$+       ��K	�`9��A�	*

logging/current_cost��;��:�+       ��K	�9��A�	*

logging/current_cost�ב;|{�0+       ��K	-�9��A�	*

logging/current_cost���;}и+       ��K	A�9��A�	*

logging/current_cost�}�;�v\+       ��K	�:��A�	*

logging/current_cost�S�;â�r+       ��K	 I:��A�	*

logging/current_cost&)�;���#+       ��K	�v:��A�	*

logging/current_cost� �;��+       ��K	E�:��A�	*

logging/current_cost�ؐ;?��z+       ��K	0�:��A�	*

logging/current_costi��;`C+       ��K	��:��A�	*

logging/current_cost���;˶ľ+       ��K	�.;��A�	*

logging/current_costg�;os��+       ��K	e];��A�
*

logging/current_costrC�;=:E+       ��K	_�;��A�
*

logging/current_cost� �;�.��+       ��K	N�;��A�
*

logging/current_cost���;h�]+       ��K	�9<��A�
*

logging/current_cost
ߏ;�Fz�+       ��K	�<��A�
*

logging/current_cost���;����+       ��K	��<��A�
*

logging/current_cost���;�)	�+       ��K	�=��A�
*

logging/current_cost���;O���+       ��K	rB=��A�
*

logging/current_cost�f�;#���+       ��K	 }=��A�
*

logging/current_costIJ�;�od�+       ��K	�=��A�
*

logging/current_cost�.�;�0+       ��K	��=��A�
*

logging/current_costj�;'�l+       ��K	&>��A�
*

logging/current_costA��;6�X+       ��K	bQ>��A�
*

logging/current_cost��;��8S+       ��K	?�>��A�
*

logging/current_costbɎ;��v�+       ��K	µ>��A�
*

logging/current_costJ��;H2�)+       ��K	R�>��A�
*

logging/current_costЙ�;�X��+       ��K	�?��A�
*

logging/current_cost-��;���+       ��K	bG?��A�
*

logging/current_cost7j�;��$+       ��K	�y?��A�
*

logging/current_costWM�;�.t+       ��K	��?��A�
*

logging/current_cost�1�;tI�,+       ��K	�?��A�
*

logging/current_cost��;t�
+       ��K	@��A�
*

logging/current_cost���;��+       ��K	<@��A�
*

logging/current_cost��;��r�+       ��K	�i@��A�
*

logging/current_costƍ;6�5.+       ��K	×@��A�
*

logging/current_cost���;��;+       ��K	x�@��A�
*

logging/current_cost˄�;��S+       ��K	�@��A�*

logging/current_cost�d�;���W+       ��K	�/A��A�*

logging/current_costE�;�H��+       ��K	�^A��A�*

logging/current_cost%�;Ͱn�+       ��K	��A��A�*

logging/current_cost<�;���+       ��K	-�A��A�*

logging/current_cost7��;�{��+       ��K	�A��A�*

logging/current_cost|ڌ;,�	 +       ��K	/B��A�*

logging/current_costnŌ;�xxB+       ��K	JIB��A�*

logging/current_cost*��;Y��+       ��K	܄B��A�*

logging/current_cost%��;;�,+       ��K	˳B��A�*

logging/current_cost���;#�rz+       ��K	E�B��A�*

logging/current_cost|v�;[T|�+       ��K	8C��A�*

logging/current_cost�c�;��@M+       ��K	�FC��A�*

logging/current_cost�Q�;��G+       ��K	k�C��A�*

logging/current_costU@�;y5�7+       ��K	֮C��A�*

logging/current_cost/�;v��+       ��K	1�C��A�*

logging/current_costI�;���U+       ��K	�D��A�*

logging/current_cost��;{�zw+       ��K	�>D��A�*

logging/current_cost���;�ŵ+       ��K	�kD��A�*

logging/current_costq�;��m+       ��K	\�D��A�*

logging/current_costߋ;���X+       ��K	3�D��A�*

logging/current_cost�ϋ;�h8�+       ��K	��D��A�*

logging/current_costS��;�HAD+       ��K		.E��A�*

logging/current_cost;�n~.+       ��K	�^E��A�*

logging/current_costޤ�;�.�?+       ��K	c�E��A�*

logging/current_cost!��;'��+       ��K	��E��A�*

logging/current_cost���;8`�{+       ��K	��E��A�*

logging/current_cost�|�;h:�+       ��K	0F��A�*

logging/current_cost�o�;�{+       ��K	�GF��A�*

logging/current_costYc�;?�w	+       ��K	vF��A�*

logging/current_cost2W�;bl��+       ��K	£F��A�*

logging/current_cost/K�;��#�+       ��K	�F��A�*

logging/current_cost}?�;vK��+       ��K	�G��A�*

logging/current_cost4�;�yHJ+       ��K	�3G��A�*

logging/current_cost-)�;w��+       ��K	ZaG��A�*

logging/current_costa�;��N+       ��K	��G��A�*

logging/current_cost��;6�A�+       ��K	>�G��A�*

logging/current_cost1	�;7f��+       ��K	7�G��A�*

logging/current_cost ��;vK�Q+       ��K	�H��A�*

logging/current_cost��;�"A+       ��K	�DH��A�*

logging/current_costY�;�i��+       ��K	�rH��A�*

logging/current_cost��;���+       ��K	~�H��A�*

logging/current_cost�؊;� �+       ��K	�H��A�*

logging/current_cost�ϊ;�"1�+       ��K	��H��A�*

logging/current_cost�Ɗ;�z>+       ��K	�,I��A�*

logging/current_cost-��;����+       ��K	^]I��A�*

logging/current_costҵ�;O�;+       ��K	�I��A�*

logging/current_cost���;D�t�+       ��K	��I��A�*

logging/current_cost���;��+       ��K	��I��A�*

logging/current_cost���;젓9+       ��K	J��A�*

logging/current_costX��;�LYF+       ��K	*CJ��A�*

logging/current_cost掊;�㲴+       ��K	�qJ��A�*

logging/current_cost���;�y+       ��K	�J��A�*

logging/current_costs��;c���+       ��K	��J��A�*

logging/current_cost�y�;TC�+       ��K	��J��A�*

logging/current_cost�r�;(��r+       ��K	M+K��A�*

logging/current_cost'l�;��h+       ��K	PXK��A�*

logging/current_cost�e�;��P�+       ��K	�K��A�*

logging/current_costg_�;-��9+       ��K	c�K��A�*

logging/current_cost'Y�;��d�+       ��K	d�K��A�*

logging/current_cost�R�;j}�@+       ��K	�L��A�*

logging/current_cost�L�;��+       ��K	!?L��A�*

logging/current_costG�;0�aJ+       ��K	mL��A�*

logging/current_cost�A�;�f�+       ��K	�L��A�*

logging/current_cost$<�;	T%+       ��K	��L��A�*

logging/current_cost�6�;�zZ�+       ��K	Y�L��A�*

logging/current_cost�1�;p���+       ��K	D%M��A�*

logging/current_cost�,�;`j��+       ��K	�RM��A�*

logging/current_cost�'�;c.�n+       ��K	�M��A�*

logging/current_cost�"�;)B�g+       ��K	J�M��A�*

logging/current_cost��;�(}�+       ��K	i�M��A�*

logging/current_costV�;u��+       ��K	?4N��A�*

logging/current_cost�;��'m+       ��K	yfN��A�*

logging/current_cost��;�{g+       ��K	�N��A�*

logging/current_cost��;W�z)+       ��K	?�N��A�*

logging/current_cost��;�$i9+       ��K	mO��A�*

logging/current_cost���;��4+       ��K	�YO��A�*

logging/current_cost���;��+       ��K	ܔO��A�*

logging/current_cost���;��+       ��K	{�O��A�*

logging/current_cost&�;rlut+       ��K	�P��A�*

logging/current_cost��;+�"E+       ��K	pEP��A�*

logging/current_cost��;� �Z+       ��K	��P��A�*

logging/current_cost��;˱�*+       ��K	��P��A�*

logging/current_cost���;E�Q�+       ��K	�P��A�*

logging/current_cost�܉;N�c�+       ��K	�=Q��A�*

logging/current_cost
ى;�2<Q+       ��K	�tQ��A�*

logging/current_cost\Չ;���+       ��K	��Q��A�*

logging/current_cost�щ;��[�+       ��K	��Q��A�*

logging/current_cost>Ή;O��C+       ��K	�R��A�*

logging/current_cost�ʉ;C��+       ��K	_;R��A�*

logging/current_costHǉ;d��+       ��K	�pR��A�*

logging/current_costQ��;4#�+       ��K	q�R��A�*

logging/current_cost���;�9+       ��K	��R��A�*

logging/current_costO��;u��+       ��K	�S��A�*

logging/current_cost�|�;�؉+       ��K	p5S��A�*

logging/current_cost|o�;,���+       ��K	�eS��A�*

logging/current_cost5d�;�Z*+       ��K	�S��A�*

logging/current_cost�Z�;ڃ�+       ��K	��S��A�*

logging/current_cost��;�,+       ��K	uT��A�*

logging/current_cost#ň;��&+       ��K	31T��A�*

logging/current_costv��;-2+       ��K	�`T��A�*

logging/current_cost�d�;�2��+       ��K	�T��A�*

logging/current_costjQ�;�u��+       ��K	��T��A�*

logging/current_cost�E�;�"��+       ��K	)�T��A�*

logging/current_costx:�;�/(+       ��K	&U��A�*

logging/current_cost}1�;����+       ��K	PSU��A�*

logging/current_costF'�;E⎖+       ��K	H�U��A�*

logging/current_costd�;�Bِ+       ��K	��U��A�*

logging/current_cost��;$k�+       ��K	H�U��A�*

logging/current_cost�	�;eLD3+       ��K	V��A�*

logging/current_cost� �;9?�+       ��K	uAV��A�*

logging/current_cost)��;�f+       ��K	!nV��A�*

logging/current_cost��;���p+       ��K	q�V��A�*

logging/current_cost��;�a��+       ��K	b�V��A�*

logging/current_cost�߇;�]�?+       ��K	��V��A�*

logging/current_costL؇;Q�N+       ��K	"(W��A�*

logging/current_cost�Ї;�s+       ��K	*UW��A�*

logging/current_cost�ɇ;k}��+       ��K	��W��A�*

logging/current_cost�;-;�M+       ��K	?�W��A�*

logging/current_cost���;��+       ��K	�W��A�*

logging/current_cost紇;�D��+       ��K	�X��A�*

logging/current_costR��;Κ�F+       ��K	J5X��A�*

logging/current_cost觇;��E�+       ��K	�bX��A�*

logging/current_cost���;h�S6+       ��K	a�X��A�*

logging/current_cost~��;P���+       ��K	��X��A�*

logging/current_cost���; ��+       ��K	�X��A�*

logging/current_cost���;2�Z�+       ��K	�Y��A�*

logging/current_cost��;Ι�+       ��K	MY��A�*

logging/current_cost���;��+       ��K	w{Y��A�*

logging/current_cost�;�F �+       ��K	ʩY��A�*

logging/current_cost�y�;��mE+       ��K	��Y��A�*

logging/current_cost�t�;���+       ��K	�
Z��A�*

logging/current_cost�o�;�w�8+       ��K	t7Z��A�*

logging/current_cost�j�;n���+       ��K	�fZ��A�*

logging/current_cost5f�;mZi�+       ��K	[�Z��A�*

logging/current_cost�a�;�m�+       ��K	s�Z��A�*

logging/current_cost�\�;�
�+       ��K	��Z��A�*

logging/current_cost�X�;PQS�+       ��K	.[��A�*

logging/current_cost�T�;G���+       ��K	�\[��A�*

logging/current_costgP�;��sI+       ��K	�[��A�*

logging/current_costxL�;����+       ��K	��[��A�*

logging/current_cost�H�;�'�%+       ��K	��[��A�*

logging/current_cost�D�;()TP+       ��K	�\��A�*

logging/current_costA�;�IB�+       ��K	�N\��A�*

logging/current_cost{=�;8��u+       ��K	�\��A�*

logging/current_cost�9�;$V{+       ��K	6�\��A�*

logging/current_cost�6�;.���+       ��K	��\��A�*

logging/current_cost*3�;�� +       ��K	�]��A�*

logging/current_cost�/�;�?Lx+       ��K	4N]��A�*

logging/current_cost�,�;>d�+       ��K	�|]��A�*

logging/current_cost�)�;h�N+       ��K	ӯ]��A�*

logging/current_costx&�;�:?�+       ��K	s�]��A�*

logging/current_costw#�;Sv��+       ��K	�^��A�*

logging/current_cost� �;z���+       ��K	�K^��A�*

logging/current_cost��;�� l+       ��K	�{^��A�*

logging/current_cost��;$�B+       ��K	}�^��A�*

logging/current_cost�;�m��+       ��K	E�^��A�*

logging/current_costd�;���+       ��K	@_��A�*

logging/current_cost��;�ɳ+       ��K	l<_��A�*

logging/current_cost�;n;v�+       ��K	�l_��A�*

logging/current_cost��;o#�c+       ��K	"�_��A�*

logging/current_cost"�;�~�+       ��K	=�_��A�*

logging/current_cost��;����+       ��K	��_��A�*

logging/current_costQ�;=GN�+       ��K	�#`��A�*

logging/current_cost�;�d8�+       ��K	kQ`��A�*

logging/current_cost��;éyI+       ��K	7�`��A�*

logging/current_cost���;'�U+       ��K	�`��A�*

logging/current_cost���;q���+       ��K	)�`��A�*

logging/current_cost���;�v+       ��K	�a��A�*

logging/current_cost���;��L+       ��K	z=a��A�*

logging/current_cost���;{�+       ��K	Voa��A�*

logging/current_cost��;$K�+       ��K	0�a��A�*

logging/current_cost1�;tlV�+       ��K	�a��A�*

logging/current_cost��;����+       ��K	��a��A�*

logging/current_cost���;[+       ��K	#b��A�*

logging/current_cost�;fb)+       ��K	>Pb��A�*

logging/current_cost{�;�!��+       ��K	�}b��A�*

logging/current_cost��;"���+       ��K	`�b��A�*

logging/current_costE�;��Ə+       ��K	��b��A�*

logging/current_cost��;d��b+       ��K	c��A�*

logging/current_cost8�;���J+       ��K	�2c��A�*

logging/current_cost��;���T+       ��K	aac��A�*

logging/current_costU�;����+       ��K	]�c��A�*

logging/current_cost��;�,O+       ��K	��c��A�*

logging/current_cost��;W��+       ��K	=�c��A�*

logging/current_cost=��;�9�+       ��K	�d��A�*

logging/current_cost�ކ;y�c+       ��K	)Ld��A�*

logging/current_cost�݆;�v�+       ��K	uzd��A�*

logging/current_costk܆;�	�;+       ��K	�d��A�*

logging/current_costDۆ;΢��+       ��K	��d��A�*

logging/current_costچ;�Ä+       ��K	;e��A�*

logging/current_cost�؆;�R�u+       ��K	s0e��A�*

logging/current_cost�׆;�cae+       ��K	^e��A�*

logging/current_cost�ֆ;R���+       ��K	��e��A�*

logging/current_cost�Ն;�8ֻ+       ��K	:�e��A�*

logging/current_cost�Ԇ;;'�t+       ��K	��e��A�*

logging/current_cost�ӆ; ~N�+       ��K	�f��A�*

logging/current_cost�҆;+� +       ��K	n@f��A�*

logging/current_cost�ц;� q�+       ��K	knf��A�*

logging/current_cost�І;�eh�+       ��K	ʢf��A�*

logging/current_cost�φ;��$,+       ��K	�f��A�*

logging/current_cost�Ά;Bf>8+       ��K	�g��A�*

logging/current_cost�͆;���+       ��K	�0g��A�*

logging/current_cost�̆;��o+       ��K	�bg��A�*

logging/current_cost̆;� �Q+       ��K	��g��A�*

logging/current_costDˆ;�li+       ��K	��g��A�*

logging/current_costwʆ;Y/?+       ��K	��g��A�*

logging/current_cost�Ɇ;e0f+       ��K	$h��A�*

logging/current_cost�Ȇ;��U2+       ��K	�Rh��A�*

logging/current_costȆ;����+       ��K	��h��A�*

logging/current_cost[ǆ;'&{+       ��K	��h��A�*

logging/current_cost�Ɔ; 47�+       ��K	x�h��A�*

logging/current_cost�ņ;b)��+       ��K	�i��A�*

logging/current_cost#ņ;g��+       ��K	�=i��A�*

logging/current_costjĆ;WsO+       ��K	2li��A�*

logging/current_cost�Æ;�0tt+       ��K	'�i��A�*

logging/current_costÆ;�uC�+       ��K	Z�i��A�*

logging/current_costm;�k��+       ��K	Y�i��A�*

logging/current_cost���;!� D+       ��K	z'j��A�*

logging/current_cost7��;��ѧ+       ��K	�Uj��A�*

logging/current_cost|��;�o��+       ��K	�j��A�*

logging/current_cost;���+       ��K	��j��A�*

logging/current_cost|��;N	��+       ��K	�j��A�*

logging/current_cost���;�$޶+       ��K	�k��A�*

logging/current_costY��;��+       ��K	sBk��A�*

logging/current_cost���;���+       ��K	iuk��A�*

logging/current_costp��;��N+       ��K	��k��A�*

logging/current_cost ��;�']C+       ��K	^�k��A�*

logging/current_cost���;:3�+       ��K	��k��A�*

logging/current_cost#��;,~�+       ��K	p/l��A�*

logging/current_costȻ�;�6Y+       ��K	U]l��A�*

logging/current_costf��;����+       ��K	.�l��A�*

logging/current_costẆ;_�`�+       ��K	x�l��A�*

logging/current_cost���;U�I�+       ��K	6�l��A�*

logging/current_cost'��;��f�+       ��K	fm��A�*

logging/current_costĹ�;��+       ��K	�Dm��A�*

logging/current_cost���;��+       ��K	�tm��A�*

logging/current_cost��;H��l+       ��K	�m��A�*

logging/current_cost��;/N_f+       ��K	��m��A�*

logging/current_cost��;l7�+       ��K	��m��A�*

logging/current_cost���;t�38+       ��K	�,n��A�*

logging/current_cost���;-l[3+       ��K	 Yn��A�*

logging/current_cost�;��+       ��K	%�n��A�*

logging/current_costʪ�;=@	+       ��K	˲n��A�*

logging/current_cost���;�9+       ��K	�n��A�*

logging/current_cost���;Ii!)+       ��K	o��A�*

logging/current_cost���;����+       ��K	@:o��A�*

logging/current_cost���;@@�+       ��K	�go��A�*

logging/current_cost>��;��~+       ��K	8�o��A�*

logging/current_costM��;�0&�+       ��K	��o��A�*

logging/current_costy��;�|�+       ��K	��o��A�*

logging/current_cost���;���N+       ��K	�p��A�*

logging/current_cost�;y �+       ��K	�Fp��A�*

logging/current_cost>��;�
��+       ��K	�tp��A�*

logging/current_cost��;����+       ��K	2�p��A�*

logging/current_costؠ�;�z��+       ��K	��p��A�*

logging/current_cost*��;�&�+       ��K	�p��A�*

logging/current_cost}��;xɭ+       ��K	D,q��A�*

logging/current_cost۞�;'u��+       ��K	.Zq��A�*

logging/current_costE��;�	K�+       ��K	Ȇq��A�*

logging/current_cost���;H�Ks+       ��K	�q��A�*

logging/current_cost��;E��+       ��K	Z�q��A�*

logging/current_cost��;���+       ��K	�r��A�*

logging/current_cost���;:�&�+       ��K	.=r��A�*

logging/current_costu��;�/�+       ��K	�jr��A�*

logging/current_cost暆;E��P+       ��K	n�r��A�*

logging/current_cost���;��L+       ��K	~�r��A�*

logging/current_cost��;�g�+       ��K	��r��A�*

logging/current_cost���;r�+       ��K	I#s��A�*

logging/current_cost��;Ey��+       ��K	mQs��A�*

logging/current_cost���;�� z+       ��K	�~s��A�*

logging/current_cost��;���+       ��K	.�s��A�*

logging/current_costӗ�;#���+       ��K	��s��A�*

logging/current_cost\��;p�UB+       ��K	�t��A�*

logging/current_costޖ�;��@+       ��K	�/t��A�*

logging/current_cost���;8(5+       ��K	�]t��A�*

logging/current_cost��;q=�+       ��K	 �t��A�*

logging/current_cost���;��+       ��K	F�t��A�*

logging/current_costU��;K�+       ��K	Y�t��A�*

logging/current_cost��;1ur+       ��K	� u��A�*

logging/current_cost���;�G�+       ��K	�Nu��A�*

logging/current_costK��;��WO+       ��K	�{u��A�*

logging/current_cost��;��Ϳ+       ��K	P�u��A�*

logging/current_cost���;[��+       ��K	��u��A�*

logging/current_costc��;�1�+       ��K	�v��A�*

logging/current_cost%��;W�"�+       ��K	C4v��A�*

logging/current_cost̒�;�-+       ��K	�cv��A�*

logging/current_cost���;8>9]+       ��K	�v��A�*

logging/current_costK��;�^}_+       ��K	�v��A�*

logging/current_cost���;��+       ��K	w�v��A�*

logging/current_cost���;G
a�+       ��K	@"w��A�*

logging/current_cost���;�[��+       ��K	mPw��A�*

logging/current_cost-��;cq�+       ��K	1~w��A�*

logging/current_cost���;���Z+       ��K	��w��A�*

logging/current_costH��;��+       ��K	>�w��A�*

logging/current_cost��;�"+       ��K	<x��A�*

logging/current_cost4��;G��+       ��K	+;x��A�*

logging/current_cost���;	��+       ��K	�lx��A�*

logging/current_cost���;p3+       ��K	��x��A�*

logging/current_costȉ�;�@�+       ��K	��x��A�*

logging/current_cost/��;^y��+       ��K	��x��A�*

logging/current_costш�;/e2{+       ��K	�&y��A�*

logging/current_cost҈�;&[$�+       ��K	Xy��A�*

logging/current_cost��;!'�+       ��K	��y��A�*

logging/current_cost*��;!(N+       ��K	��y��A�*

logging/current_cost��;�
�+       ��K	��y��A�*

logging/current_cost��;�E�+       ��K	Bz��A�*

logging/current_cost���;m�+       ��K	�=z��A�*

logging/current_cost���;.���+       ��K	xkz��A�*

logging/current_cost���;-SH+       ��K	��z��A�*

logging/current_cost��;樾�+       ��K	�z��A�*

logging/current_cost]��;l8�+       ��K	u�z��A�*

logging/current_cost���;+�j�+       ��K	�!{��A�*

logging/current_costs��;��%�+       ��K	EM{��A�*

logging/current_coste��;��7�+       ��K	;�{��A�*

logging/current_cost냆;�8��+       ��K	�{��A�*

logging/current_cost���;'V��+       ��K	.�{��A�*

logging/current_cost��;�"�:+       ��K	�|��A�*

logging/current_cost䂆;����+       ��K	4L|��A�*

logging/current_cost*��;1Qe+       ��K	�z|��A�*

logging/current_costV��;�u�}+       ��K	��|��A�*

logging/current_cost���;�2�+       ��K	��|��A�*

logging/current_costW��;�]�+       ��K	� }��A�*

logging/current_cost��;��M+       ��K	�.}��A�*

logging/current_cost��;���+       ��K	$[}��A�*

logging/current_cost;��;�mǪ+       ��K	��}��A�*

logging/current_cost{��;�6Π+       ��K	W�}��A�*

logging/current_cost���;ﱒT+       ��K	�}��A�*

logging/current_cost���;\!}�+       ��K	�~��A�*

logging/current_cost���;͡p�+       ��K	�E~��A�*

logging/current_costւ�;ٖ6�+       ��K	5r~��A�*

logging/current_cost��;�7�+       ��K	3�~��A�*

logging/current_cost逆;��8+       ��K	��~��A�*

logging/current_cost���;f�+       ��K	u�~��A�*

logging/current_cost�;��}�+       ��K	f'��A�*

logging/current_costb��;�t�+       ��K	�R��A�*

logging/current_cost���; X��+       ��K	+���A�*

logging/current_costn��;'�J�+       ��K	ۮ��A�*

logging/current_costπ�;T�5+       ��K	����A�*

logging/current_cost���;�A�+       ��K	p���A�*

logging/current_costi��;��W�+       ��K	�9���A�*

logging/current_cost��;�&�]+       ��K	�e���A�*

logging/current_cost��;��+       ��K	�����A�*

logging/current_cost��;
��+       ��K	�����A�*

logging/current_cost!��;N~o+       ��K	�A�*

logging/current_cost���;B�+       ��K	u���A�*

logging/current_cost��;Gi(�+       ��K	G���A�*

logging/current_cost���;ؖ�/+       ��K	�t���A�*

logging/current_cost�;Q���+       ��K	�����A�*

logging/current_cost뀆;�:�R+       ��K	�ց��A�*

logging/current_cost���;�F�h+       ��K	����A�*

logging/current_cost,��;t{^�+       ��K	Z0���A�*

logging/current_cost6�;}�!+       ��K	�^���A�*

logging/current_costQ�;('�+       ��K	L����A�*

logging/current_costg�;$4}S+       ��K	'����A�*

logging/current_costc�;���N+       ��K	]낱�A�*

logging/current_cost��;�{��+       ��K	���A�*

logging/current_cost��;۴�+       ��K	�H���A�*

logging/current_cost�~�;QZ�+       ��K	�w���A�*

logging/current_cost\��;��7�+       ��K	b����A�*

logging/current_costk�;{�
�+       ��K	�܃��A�*

logging/current_costg~�;��Ls+       ��K	K���A�*

logging/current_cost��;)Jȍ+       ��K	-:���A�*

logging/current_cost�~�;0�o�+       ��K	Rg���A�*

logging/current_cost�~�;��	�+       ��K	����A�*

logging/current_cost�~�;9��+       ��K	ń��A�*

logging/current_costw~�;:���+       ��K	p��A�*

logging/current_cost[~�;Dhj�+       ��K	%"���A�*

logging/current_cost�~�;��L+       ��K	�U���A�*

logging/current_cost�;v�r,+       ��K	����A�*

logging/current_cost ~�;�z+       ��K	2����A�*

logging/current_cost��;J�+       ��K	܅��A�*

logging/current_costn�;5N��+       ��K	����A�*

logging/current_cost~�;��+       ��K	�;���A�*

logging/current_cost�}�;u&�+       ��K	�i���A�*

logging/current_cost�~�;L�u�+       ��K	�����A�*

logging/current_cost�~�;�1��+       ��K	ǆ��A�*

logging/current_cost�}�;�.��+       ��K	�����A�*

logging/current_cost~�;[��+       ��K	�%���A�*

logging/current_cost_~�;lN�Y+       ��K	'U���A�*

logging/current_cost�}�;���+       ��K	X����A�*

logging/current_costw}�;H>�+       ��K	1߇��A�*

logging/current_cost�}�;�Ȱ+       ��K	�-���A�*

logging/current_costp}�;YJ�F+       ��K	�x���A�*

logging/current_costy}�;=׻1+       ��K	C����A�*

logging/current_cost<}�;묖9+       ��K	�����A�*

logging/current_cost�}�;��;+       ��K	�;���A�*

logging/current_costQ}�;�'(�+       ��K	Ov���A�*

logging/current_cost}�;��@+       ��K	 ����A�*

logging/current_costs}�;VH�+       ��K	�A�*

logging/current_cost}�;���q+       ��K	!���A�*

logging/current_costA}�;����+       ��K	�Y���A�*

logging/current_cost}�;}�l)+       ��K	Џ���A�*

logging/current_cost�|�;�F�+       ��K	=Ê��A�*

logging/current_cost�|�;���+       ��K	P����A�*

logging/current_cost}�;���V+       ��K	�+���A�*

logging/current_cost9}�;R�ҧ+       ��K	e���A�*

logging/current_cost�|�;��ޓ+       ��K	����A�*

logging/current_cost�~�;�]��+       ��K	Qԋ��A�*

logging/current_cost$}�;�G�6+       ��K	����A�*

logging/current_cost�|�;�o�+       ��K	1?���A�*

logging/current_cost*}�;Y�[+       ��K	u���A�*

logging/current_costx|�;{��+       ��K	�����A�*

logging/current_cost�|�;�*�+       ��K	�ӌ��A�*

logging/current_cost�|�;27�9+       ��K	����A�*

logging/current_cost�|�;h��+       ��K	9���A�*

logging/current_costp~�;��=�+       ��K	�g���A�*

logging/current_cost>}�;�O5+       ��K	�����A�*

logging/current_cost+}�;�E�+       ��K	�č��A�*

logging/current_cost�|�;�E�+       ��K	>��A�*

logging/current_cost~�;��+       ��K	�)���A�*

logging/current_cost`|�;�CX+       ��K	pe���A�*

logging/current_cost,}�;�0��+       ��K	�����A�*

logging/current_cost�|�;�l��+       ��K	�Ў��A�*

logging/current_cost�|�;&d7�+       ��K	� ���A�*

logging/current_costx|�;b���+       ��K	�.���A�*

logging/current_cost�|�;K�z+       ��K	�\���A�*

logging/current_cost|�;�@��+       ��K	$����A�*

logging/current_cost�|�;x�?�+       ��K	5����A�*

logging/current_cost�}�;����+       ��K	[叱�A�*

logging/current_cost�{�;����+       ��K	���A�*

logging/current_cost�}�;QM�+       ��K	D���A�*

logging/current_cost�|�;�L�J+       ��K	�s���A�*

logging/current_cost|�;cEX+       ��K	�����A�*

logging/current_cost�|�;��_�+       ��K	pѐ��A�*

logging/current_cost�{�;� �+       ��K	���A�*

logging/current_cost�{�;>W8�+       ��K	/���A�*

logging/current_cost7}�;���+       ��K	�]���A�*

logging/current_costa|�;�%e�+       ��K	�����A�*

logging/current_costb|�;8�M+       ��K	�����A�*

logging/current_cost�|�;��U'+       ��K		��A�*

logging/current_cost�{�;�BXY+       ��K	�+���A�*

logging/current_cost�|�;^ļ+       ��K	�]���A�*

logging/current_cost|�;>,�+       ��K	����A�*

logging/current_costz{�;j�{�+       ��K	9����A�*

logging/current_cost|�;0�y�+       ��K	k钱�A�*

logging/current_cost�{�;b�+       ��K	-���A�*

logging/current_cost�{�;v�g+       ��K	!J���A�*

logging/current_cost�{�;��+       ��K	sx���A�*

logging/current_cost=}�;g�Y�+       ��K	�����A�*

logging/current_costn|�;��L�+       ��K	�ޓ��A�*

logging/current_costh{�;���+       ��K	����A�*

logging/current_costN{�;[U/n+       ��K	�<���A�*

logging/current_cost�{�;.��+       ��K	�j���A�*

logging/current_cost�{�;,��-+       ��K	ئ���A�*

logging/current_cost�{�;0oH�+       ��K	�Ք��A�*

logging/current_cost�{�;M/��+       ��K	���A�*

logging/current_cost8{�;\Ď�+       ��K	�3���A�*

logging/current_cost{�;�=\+       ��K	�a���A�*

logging/current_cost>{�;Y�2+       ��K	!����A�*

logging/current_costn{�;xۘ:+       ��K	���A�*

logging/current_cost�z�;;y�f+       ��K	��A�*

logging/current_cost�{�;�u7-+       ��K	����A�*

logging/current_cost�{�;���+       ��K	�K���A�*

logging/current_cost;{�;v��h+       ��K	�z���A�*

logging/current_cost�z�;����+       ��K	�����A�*

logging/current_cost4{�;m��+       ��K	&Ֆ��A�*

logging/current_cost{�;��w3+       ��K	����A�*

logging/current_costU{�;�	-�+       ��K	�2���A�*

logging/current_cost�{�;��n+       ��K	_���A�*

logging/current_costE{�;݀%S+       ��K	)����A�*

logging/current_cost�z�;�+       ��K	`����A�*

logging/current_cost{�;�{R+       ��K	�旱�A�*

logging/current_cost	{�;Xn$�+       ��K	���A�*

logging/current_cost�z�;ZAʮ+       ��K	@���A�*

logging/current_cost�z�;��']+       ��K	n���A�*

logging/current_cost�z�;K�d�+       ��K	Λ���A�*

logging/current_costsz�;�=1�+       ��K	0͘��A�*

logging/current_costI{�;��+       ��K	�����A�*

logging/current_cost�z�;8�܂+       ��K	�,���A�*

logging/current_cost{�;&ЎN+       ��K	�\���A�*

logging/current_cost�z�;��J�+       ��K	�����A�*

logging/current_cost�{�;$X�+       ��K	(����A�*

logging/current_cost�z�;�|�B+       ��K	>癱�A�*

logging/current_cost�z�;�H3+       ��K	����A�*

logging/current_costm{�;��+       ��K	�A���A�*

logging/current_cost�z�;�X�+       ��K	�p���A�*

logging/current_costDz�;�&eS+       ��K	����A�*

logging/current_cost�z�;�oT?+       ��K	�˚��A�*

logging/current_cost]z�; ���+       ��K	����A�*

logging/current_cost}z�;����+       ��K	�*���A�*

logging/current_cost�z�;U'_`+       ��K	�U���A�*

logging/current_cost-z�;:�eG+       ��K	Ǉ���A�*

logging/current_cost�z�;�>i�+       ��K	�����A�*

logging/current_cost�{�;���O+       ��K	�雱�A�*

logging/current_costSz�;�^&+       ��K	����A�*

logging/current_cost�|�;��m+       ��K	�H���A�*

logging/current_costC{�;U�Ψ+       ��K	�|���A�*

logging/current_costz�;�P+       ��K	Ѫ���A�*

logging/current_cost�{�;�L@�+       ��K	�՜��A�*

logging/current_cost�|�;3�+       ��K	r���A�*

logging/current_cost�y�;{k0�+       ��K	9���A�*

logging/current_costp{�;�T+       ��K	�i���A�*

logging/current_cost_{�;7c_�+       ��K	����A�*

logging/current_cost�y�;X�q+       ��K	Q̝��A�*

logging/current_cost	z�;!ub�+       ��K	�����A�*

logging/current_costyz�;TY�+       ��K	�.���A�*

logging/current_cost�z�;��Ww+       ��K	�`���A�*

logging/current_cost0z�;aQ[+       ��K	ᑞ��A�*

logging/current_costjy�;&�i+       ��K	�����A�*

logging/current_cost]z�;ށ}�+       ��K	�잱�A�*

logging/current_cost�z�;}�{+       ��K	����A�*

logging/current_cost�y�;؋�K+       ��K	�M���A�*

logging/current_cost{�;Sw�	+       ��K	�|���A�*

logging/current_cost�z�;�P$�+       ��K	@����A�*

logging/current_costCz�;��Z;+       ��K	�֟��A�*

logging/current_cost�z�;-+/+       ��K	����A�*

logging/current_cost�y�;n�Y�+       ��K	n4���A�*

logging/current_cost�z�;����+       ��K	/c���A�*

logging/current_costKz�;zk�+       ��K	�����A�*

logging/current_cost�y�;��<�+       ��K	X����A�*

logging/current_cost�y�;I~vk+       ��K	�A�*

logging/current_costz�;����+       ��K	$���A�*

logging/current_cost�y�;���+       ��K	�I���A�*

logging/current_cost�y�;b�-t+       ��K	x����A�*

logging/current_cost�y�;��+       ��K	�����A�*

logging/current_costvy�;��+       ��K	Cۡ��A�*

logging/current_costqy�; ��\+       ��K	���A�*

logging/current_costIy�;]o�+       ��K	�3���A�*

logging/current_cost�y�;��>+       ��K	�`���A�*

logging/current_cost%y�;	��6+       ��K	k����A�*

logging/current_costy�;xUe�+       ��K	����A�*

logging/current_costy�;{�ئ+       ��K	�뢱�A�*

logging/current_cost�x�;��<H+       ��K	r���A�*

logging/current_cost�y�;}�B+       ��K	�E���A�*

logging/current_cost�y�;|��+       ��K	ss���A�*

logging/current_cost�x�;�L��+       ��K	x����A�*

logging/current_cost2y�;P��G+       ��K	XΣ��A�*

logging/current_costgy�;`���+       ��K	�����A�*

logging/current_cost�x�;45+       ��K	j)���A�*

logging/current_cost	{�;ZB��+       ��K	W���A�*

logging/current_cost@|�;��_�+       ��K	0����A�*

logging/current_cost�y�;�K7�+       ��K	�����A�*

logging/current_cost�y�;���+       ��K	�ᤱ�A�*

logging/current_cost�z�;���+       ��K	����A�*

logging/current_cost�y�;}g��+       ��K	?���A�*

logging/current_cost�x�;�TJ�+       ��K	:l���A�*

logging/current_cost�y�;@�8+       ��K	ϙ���A�*

logging/current_costpy�;U�wm+       ��K	�ƥ��A� *

logging/current_cost*y�;=R^x+       ��K	~����A� *

logging/current_cost�y�;�Ed+       ��K	�"���A� *

logging/current_cost�x�;�`p�+       ��K	'O���A� *

logging/current_costTy�;�y�M+       ��K	�~���A� *

logging/current_costz�;}�+       ��K	�����A� *

logging/current_cost8y�;�C�+       ��K	/ߦ��A� *

logging/current_cost�x�;���+       ��K	`���A� *

logging/current_cost�y�;��1�+       ��K	�B���A� *

logging/current_cost.y�;:\�+       ��K	p���A� *

logging/current_cost�x�;�t'+       ��K	�����A� *

logging/current_cost�y�;�n��+       ��K	�˧��A� *

logging/current_cost�x�;8r��+       ��K	�����A� *

logging/current_cost!y�;�.�+       ��K	I*���A� *

logging/current_cost�y�;K?sP+       ��K	�X���A� *

logging/current_cost;z�;)M�;+       ��K	{����A� *

logging/current_cost�x�;�E�+       ��K	�����A� *

logging/current_costy�;Z�j�+       ��K	f쨱�A� *

logging/current_costXy�;�4�+       ��K	����A� *

logging/current_costHy�;Y
i�+       ��K	�E���A� *

logging/current_cost�x�;W�-}+       ��K	�x���A� *

logging/current_cost�y�;��N�+       ��K	ݧ���A� *

logging/current_cost\y�;��Њ+       ��K	�ԩ��A� *

logging/current_cost�x�;�Ρ+       ��K	����A� *

logging/current_cost�x�;����+       ��K	;7���A� *

logging/current_cost�x�;��9
+       ��K	�g���A�!*

logging/current_costux�;z�d+       ��K	𖪱�A�!*

logging/current_cost�x�;)}'+       ��K	�Ī��A�!*

logging/current_costfy�;��d+       ��K	�����A�!*

logging/current_cost�x�;xPS+       ��K	2#���A�!*

logging/current_cost�x�;�V+       ��K	nQ���A�!*

logging/current_costy�;gĜ�+       ��K	|����A�!*

logging/current_costBy�;a��D+       ��K	�����A�!*

logging/current_cost�x�;���+       ��K	�ޫ��A�!*

logging/current_cost�x�;ek�+       ��K	���A�!*

logging/current_cost�x�;б��+       ��K	�;���A�!*

logging/current_costQx�;���+       ��K	�k���A�!*

logging/current_cost�x�;m��+       ��K	�����A�!*

logging/current_cost�x�;f�1+       ��K	vƬ��A�!*

logging/current_cost�x�;��z+       ��K	!����A�!*

logging/current_cost�x�;|�ƙ+       ��K	Z#���A�!*

logging/current_cost+x�;;�+       ��K	{S���A�!*

logging/current_costhx�;��5z+       ��K	�����A�!*

logging/current_costtx�;��;�+       ��K	خ���A�!*

logging/current_costbx�;k��+       ��K	bܭ��A�!*

logging/current_cost^x�;�ϡ[+       ��K	�
���A�!*

logging/current_costax�;�\_+       ��K	�9���A�!*

logging/current_cost�x�;|Ö�+       ��K	u���A�!*

logging/current_costEx�;�yp/+       ��K	إ���A�!*

logging/current_costx�;}
�?+       ��K	ծ��A�!*

logging/current_cost�x�;p=+       ��K	R���A�!*

logging/current_cost�x�;��;+       ��K	�.���A�"*

logging/current_costRx�;�V�I+       ��K	.\���A�"*

logging/current_costux�;�f�]+       ��K	׉���A�"*

logging/current_cost;x�;���+       ��K	{����A�"*

logging/current_cost�x�;O͂]+       ��K	�㯱�A�"*

logging/current_costhy�;�|+       ��K	����A�"*

logging/current_cost�x�;�tG+       ��K	n=���A�"*

logging/current_costz�;�z�>+       ��K	�j���A�"*

logging/current_cost6{�;|Z�4+       ��K	�����A�"*

logging/current_costYx�;nL�+       ��K	�ǰ��A�"*

logging/current_cost$y�;8�bQ+       ��K	&����A�"*

logging/current_cost*x�;ٗZ�+       ��K	q#���A�"*

logging/current_cost�x�;��ZR+       ��K	�O���A�"*

logging/current_cost�x�;�_+       ��K	�|���A�"*

logging/current_cost�x�;�J�z+       ��K	T����A�"*

logging/current_costx�;�:te+       ��K	�ر��A�"*

logging/current_cost6x�;Zn��+       ��K		���A�"*

logging/current_costIx�;gjm+       ��K	�6���A�"*

logging/current_cost@y�;{<1S+       ��K	�d���A�"*

logging/current_cost�x�;?���+       ��K	ސ���A�"*

logging/current_costPz�;4Y%�+       ��K	Z����A�"*

logging/current_cost1x�;�!'�+       ��K	겱�A�"*

logging/current_cost�x�;K�+       ��K	x���A�"*

logging/current_cost�x�;)��+       ��K	�E���A�"*

logging/current_costPx�;�m�3+       ��K	lr���A�"*

logging/current_cost�w�;��j3+       ��K	�����A�#*

logging/current_cost�y�;O��+       ��K	˳��A�#*

logging/current_costJx�;v	��+       ��K	H����A�#*

logging/current_costfy�;�>�t+       ��K	'���A�#*

logging/current_cost}x�;��]�+       ��K	�U���A�#*

logging/current_costr{�;@E��+       ��K	�����A�#*

logging/current_cost�w�;G���+       ��K	b����A�#*

logging/current_cost�w�;�d��+       ��K	Nᴱ�A�#*

logging/current_cost�x�;�,�+       ��K	0���A�#*

logging/current_cost�x�;q��+       ��K	�=���A�#*

logging/current_costRx�;p�"2+       ��K	�j���A�#*

logging/current_costx�;��W+       ��K	ٚ���A�#*

logging/current_cost�w�;UP+       ��K	˵��A�#*

logging/current_costJx�;8�[+       ��K	�����A�#*

logging/current_cost�w�;o�3�+       ��K	(���A�#*

logging/current_cost x�;�Ts+       ��K	�T���A�#*

logging/current_cost�w�;gQ.%+       ��K	솶��A�#*

logging/current_cost[x�;�
�+       ��K	����A�#*

logging/current_cost�x�;w>�q+       ��K	�䶱�A�#*

logging/current_cost�w�;6�q+       ��K	`���A�#*

logging/current_cost4x�;Z/X�+       ��K	\F���A�#*

logging/current_cost�x�;�T�+       ��K	�t���A�#*

logging/current_cost_y�;���d+       ��K	�����A�#*

logging/current_costEx�;Pf&�+       ��K	hз��A�#*

logging/current_cost?y�;�B�+       ��K	h���A�#*

logging/current_costx�;�>�+       ��K	�0���A�#*

logging/current_cost�w�;��+       ��K	�\���A�$*

logging/current_costvz�;`(~�+       ��K	r����A�$*

logging/current_cost�x�;Y��+       ��K	�����A�$*

logging/current_cost�x�;5 �+       ��K	����A�$*

logging/current_costnx�;�v3W+       ��K	n5���A�$*

logging/current_cost?y�;09 h+       ��K	/k���A�$*

logging/current_cost*x�;���+       ��K	�����A�$*

logging/current_cost7x�;���+       ��K	EϹ��A�$*

logging/current_cost0x�;T1C�+       ��K	g���A�$*

logging/current_costx�;��+       ��K	�;���A�$*

logging/current_costFx�;fb<+       ��K	�q���A�$*

logging/current_cost+x�;I[��+       ��K	=����A�$*

logging/current_cost�y�;�_��+       ��K	�׺��A�$*

logging/current_cost5y�;�Qb+       ��K	����A�$*

logging/current_cost�w�;��V�+       ��K	%A���A�$*

logging/current_costjx�;�p��+       ��K	����A�$*

logging/current_costDy�;lRi+       ��K	YԻ��A�$*

logging/current_costx�;�ʬ+       ��K	����A�$*

logging/current_cost?x�;��T�+       ��K	f���A�$*

logging/current_cost`y�;�M�+       ��K	槼��A�$*

logging/current_cost�w�;n�ڈ+       ��K	b鼱�A�$*

logging/current_cost�w�;��#+       ��K	?)���A�$*

logging/current_cost7x�;�2O+       ��K	ze���A�$*

logging/current_costtx�;CX��+       ��K	z����A�$*

logging/current_cost7x�;u{�1+       ��K	]׽��A�$*

logging/current_cost�w�;4̄+       ��K	���A�$*

logging/current_cost�w�;l��+       ��K	�P���A�%*

logging/current_cost!x�;R?l>+       ��K	�����A�%*

logging/current_cost�w�;7aN+       ��K	�پ��A�%*

logging/current_costx�;�|O�+       ��K	���A�%*

logging/current_cost�w�;;Cc@+       ��K	aB���A�%*

logging/current_cost�w�;
�4+       ��K	{x���A�%*

logging/current_cost�w�;���+       ��K	M����A�%*

logging/current_cost�w�;�\!+       ��K	�ܿ��A�%*

logging/current_cost�w�;H��3+       ��K	[���A�%*

logging/current_cost�w�;�E^+       ��K	v=���A�%*

logging/current_costnx�;�r�+       ��K	It���A�%*

logging/current_cost�w�;[7A+       ��K	�����A�%*

logging/current_cost�x�;V�=}+       ��K	�����A�%*

logging/current_cost�x�;n�MB+       ��K	����A�%*

logging/current_costx�;��>+       ��K	�}���A�%*

logging/current_cost[x�;�R+       ��K	�����A�%*

logging/current_cost2x�;���d+       ��K	�±�A�%*

logging/current_cost.x�;O/�7+       ��K	U±�A�%*

logging/current_costx�;8��?+       ��K	B�±�A�%*

logging/current_cost�w�;8W�v+       ��K	�±�A�%*

logging/current_cost�w�;5�W+       ��K	f9ñ�A�%*

logging/current_cost�w�;lr�+       ��K	�pñ�A�%*

logging/current_costuw�;"{[+       ��K	��ñ�A�%*

logging/current_cost�w�; �+       ��K	u�ñ�A�%*

logging/current_cost x�;���+       ��K	u'ı�A�%*

logging/current_cost�w�;K�+       ��K	�rı�A�&*

logging/current_cost�w�;5U{+       ��K	g�ı�A�&*

logging/current_cost�w�;b�0+       ��K	��ı�A�&*

logging/current_cost�x�;���R+       ��K	q0ű�A�&*

logging/current_costcw�;*��+       ��K	�eű�A�&*

logging/current_costjx�;�W)+       ��K	�ű�A�&*

logging/current_cost�w�;��	+       ��K	?�ű�A�&*

logging/current_costx�;��+       ��K	Ʊ�A�&*

logging/current_cost�w�;rVî+       ��K	)PƱ�A�&*

logging/current_costnx�;p�̊+       ��K	܁Ʊ�A�&*

logging/current_cost�w�;�Q��+       ��K	��Ʊ�A�&*

logging/current_cost�y�;	/�+       ��K	��Ʊ�A�&*

logging/current_cost�w�;��h�+       ��K	lǱ�A�&*

logging/current_costx�;˵k+       ��K	�HǱ�A�&*

logging/current_costpw�;�+�+       ��K	�wǱ�A�&*

logging/current_cost�w�;��"'+       ��K	B�Ǳ�A�&*

logging/current_cost�w�;=�&�+       ��K	9�Ǳ�A�&*

logging/current_cost�w�;D�b+       ��K	�ȱ�A�&*

logging/current_cost�w�;�jh�+       ��K	Iȱ�A�&*

logging/current_cost�w�;��"+       ��K	�ȱ�A�&*

logging/current_costmw�;��+       ��K	 �ȱ�A�&*

logging/current_cost�w�;( N+       ��K	��ȱ�A�&*

logging/current_costx�;��t+       ��K	7ɱ�A�&*

logging/current_cost<w�;7��+       ��K	,Iɱ�A�&*

logging/current_costw�;K���+       ��K	Nvɱ�A�&*

logging/current_cost�w�;�Rg.+       ��K	��ɱ�A�&*

logging/current_cost�w�;���N+       ��K	��ɱ�A�'*

logging/current_cost�w�;Q�Z+       ��K	��ɱ�A�'*

logging/current_costhw�;�i�Q+       ��K	�,ʱ�A�'*

logging/current_cost�w�;��-+       ��K	�\ʱ�A�'*

logging/current_costpw�;��=�+       ��K	��ʱ�A�'*

logging/current_cost�w�;w%U+       ��K	��ʱ�A�'*

logging/current_cost�w�;�#�+       ��K	q�ʱ�A�'*

logging/current_cost�w�;hz��+       ��K	9˱�A�'*

logging/current_cost�w�; �7+       ��K	8E˱�A�'*

logging/current_cost�w�;9У+       ��K	�v˱�A�'*

logging/current_cost�w�;��kv+       ��K	��˱�A�'*

logging/current_costhw�;�V"+       ��K	�˱�A�'*

logging/current_cost�x�;Qe�Q+       ��K	]̱�A�'*

logging/current_costcw�;�o�d+       ��K	:0̱�A�'*

logging/current_costdw�;|G@+       ��K	�]̱�A�'*

logging/current_cost�w�;��0"+       ��K	��̱�A�'*

logging/current_costLw�;�;��+       ��K	��̱�A�'*

logging/current_cost?w�;^Y�+       ��K	��̱�A�'*

logging/current_cost�w�;*� +       ��K	�ͱ�A�'*

logging/current_costSw�;��'+       ��K	�Dͱ�A�'*

logging/current_costJw�;q�� +       ��K	�rͱ�A�'*

logging/current_cost�w�;\>+       ��K	��ͱ�A�'*

logging/current_costow�;;7�@+       ��K	�ͱ�A�'*

logging/current_cost�w�;��&�+       ��K	��ͱ�A�'*

logging/current_cost�w�;r2�r+       ��K	�1α�A�'*

logging/current_cost�w�;�Nb+       ��K	�eα�A�(*

logging/current_cost�w�;x�I�+       ��K	e�α�A�(*

logging/current_cost�w�;aҡF+       ��K	�α�A�(*

logging/current_cost�w�;��+       ��K	��α�A�(*

logging/current_costpw�;YK8�+       ��K	�ϱ�A�(*

logging/current_costFx�;���+       ��K	�Qϱ�A�(*

logging/current_cost�w�;m���+       ��K	,�ϱ�A�(*

logging/current_costux�;�K5�+       ��K	��ϱ�A�(*

logging/current_cost�w�;�ۿ^+       ��K	��ϱ�A�(*

logging/current_cost�w�;�+��+       ��K	б�A�(*

logging/current_costHw�;1�vP+       ��K	6?б�A�(*

logging/current_cost|w�;����+       ��K	�lб�A�(*

logging/current_cost�x�;�.��+       ��K	6�б�A�(*

logging/current_cost�w�;"�2+       ��K	��б�A�(*

logging/current_cost�x�;ƴ��+       ��K	��б�A�(*

logging/current_cost�w�;30Έ+       ��K	�*ѱ�A�(*

logging/current_costQw�;
��+       ��K	�Yѱ�A�(*

logging/current_costbx�;�^>+       ��K	S�ѱ�A�(*

logging/current_cost�w�;;�<�+       ��K	d�ѱ�A�(*

logging/current_cost�w�;)
$�+       ��K	��ѱ�A�(*

logging/current_cost
x�;���+       ��K	 ұ�A�(*

logging/current_cost�w�;~c�+       ��K	CMұ�A�(*

logging/current_cost�w�;��+       ��K	/�ұ�A�(*

logging/current_costTx�;���+       ��K	�ұ�A�(*

logging/current_cost9w�;�(ro+       ��K	��ұ�A�(*

logging/current_cost�w�;�!+       ��K	 ӱ�A�(*

logging/current_costMw�;|-X+       ��K	)?ӱ�A�)*

logging/current_cost'w�;N�@�+       ��K	�nӱ�A�)*

logging/current_cost�w�;7�E+       ��K	�ӱ�A�)*

logging/current_cost�w�;K��+       ��K	��ӱ�A�)*

logging/current_cost?w�;C�+       ��K	Ա�A�)*

logging/current_cost�w�;I%U+       ��K	�5Ա�A�)*

logging/current_costtw�;��+"+       ��K	5cԱ�A�)*

logging/current_costx�;��p+       ��K	v�Ա�A�)*

logging/current_cost$w�;ٸ�	+       ��K	^�Ա�A�)*

logging/current_costAw�;���+       ��K	�Ա�A�)*

logging/current_cost�w�;���+       ��K	=ձ�A�)*

logging/current_costJw�;��y+       ��K	�Mձ�A�)*

logging/current_cost�w�;��і+       ��K	�{ձ�A�)*

logging/current_cost;w�;Ko�%+       ��K	��ձ�A�)*

logging/current_costw�;�B�+       ��K	�ձ�A�)*

logging/current_cost�w�;Ձy�+       ��K	Jֱ�A�)*

logging/current_costy�;�Ŷd+       ��K	�2ֱ�A�)*

logging/current_cost�w�;��r+       ��K	�bֱ�A�)*

logging/current_cost�w�;��+       ��K	��ֱ�A�)*

logging/current_cost|w�;�^�+       ��K	w�ֱ�A�)*

logging/current_cost w�;�^\�+       ��K	��ֱ�A�)*

logging/current_costw�;��Ԓ+       ��K	�ױ�A�)*

logging/current_cost#w�;��q+       ��K	Lױ�A�)*

logging/current_costZx�;R5� +       ��K	�yױ�A�)*

logging/current_costwx�;�~�n+       ��K	t�ױ�A�)*

logging/current_costUx�;n`��+       ��K	v�ױ�A�)*

logging/current_cost�x�;(j� +       ��K	ر�A�**

logging/current_cost�w�;�	jQ+       ��K	�1ر�A�**

logging/current_costx�;���0+       ��K	:_ر�A�**

logging/current_cost�w�;���+       ��K	 �ر�A�**

logging/current_cost�w�;��̛+       ��K	��ر�A�**

logging/current_costhw�;�)��+       ��K	��ر�A�**

logging/current_costx�;���+       ��K	Iٱ�A�**

logging/current_costgx�;���+       ��K	�Fٱ�A�**

logging/current_costjw�;�s3"+       ��K	�sٱ�A�**

logging/current_costw�;_�+       ��K	��ٱ�A�**

logging/current_cost�w�;I$��+       ��K	��ٱ�A�**

logging/current_cost�x�;V�'&+       ��K	P�ٱ�A�**

logging/current_cost�w�;���+       ��K	s*ڱ�A�**

logging/current_cost�w�;��8�+       ��K	>Xڱ�A�**

logging/current_cost�w�;�Ĉ�+       ��K	ʆڱ�A�**

logging/current_cost[w�;�3f�+       ��K	~�ڱ�A�**

logging/current_costw�;�"˧+       ��K	E�ڱ�A�**

logging/current_cost�w�;1ʶ+       ��K	V۱�A�**

logging/current_costRw�;�Dm+       ��K	x;۱�A�**

logging/current_cost�w�;�X�l+       ��K	�i۱�A�**

logging/current_cost�w�;|q٨+       ��K	�۱�A�**

logging/current_costIw�;k���+       ��K	��۱�A�**

logging/current_cost�w�;^�a�+       ��K	o�۱�A�**

logging/current_cost�x�;˛#,+       ��K	Aܱ�A�**

logging/current_costdx�;�-+       ��K	JQܱ�A�**

logging/current_costlx�;��2@+       ��K	��ܱ�A�+*

logging/current_cost
x�;���+       ��K	<�ܱ�A�+*

logging/current_costnx�;�	s�+       ��K	��ܱ�A�+*

logging/current_cost�w�;<2�c+       ��K	ݱ�A�+*

logging/current_costWw�;���L+       ��K	-Dݱ�A�+*

logging/current_cost{�;�R�b+       ��K	rݱ�A�+*

logging/current_cost�y�;6~Ғ+       ��K	բݱ�A�+*

logging/current_costnw�;-!8�+       ��K	S�ݱ�A�+*

logging/current_costSx�;�lO+       ��K	ޱ�A�+*

logging/current_costFw�;�81�+       ��K	�6ޱ�A�+*

logging/current_cost-w�;Qz�+       ��K	dgޱ�A�+*

logging/current_cost w�;G��+       ��K	��ޱ�A�+*

logging/current_cost�x�;�<"+       ��K	V�ޱ�A�+*

logging/current_cost�w�;T'�+       ��K	��ޱ�A�+*

logging/current_cost�w�;6*j+       ��K	c"߱�A�+*

logging/current_costx�;�+       ��K	JS߱�A�+*

logging/current_cost�v�;���+       ��K	�߱�A�+*

logging/current_cost�w�;M+bT+       ��K	��߱�A�+*

logging/current_cost�x�;o��+       ��K	i�߱�A�+*

logging/current_costx�;�>�+       ��K	���A�+*

logging/current_cost�w�;B�a+       ��K	>=��A�+*

logging/current_cost%x�;NQHp+       ��K	�j��A�+*

logging/current_costzx�;��;�+       ��K	����A�+*

logging/current_costDw�;�}6+       ��K	|���A�+*

logging/current_cost�w�;o,3P+       ��K	Q���A�+*

logging/current_costBz�;Q8?+       ��K	-��A�+*

logging/current_cost�y�;3��!+       ��K	�Z��A�,*

logging/current_costQw�;آߎ+       ��K	����A�,*

logging/current_cost�y�;
���+       ��K	n���A�,*

logging/current_costy�;�V�+       ��K	����A�,*

logging/current_cost.x�;K��+       ��K	���A�,*

logging/current_cost�|�;�Q (+       ��K	�@��A�,*

logging/current_cost�w�;S��a+       ��K	�o��A�,*

logging/current_cost�z�;�R�+       ��K	����A�,*

logging/current_cost�x�;=Q�+       ��K	M���A�,*

logging/current_cost�w�;����+       ��K	���A�,*

logging/current_cost#x�;��ڤ+       ��K	V&��A�,*

logging/current_cost�x�;MN+       ��K	�S��A�,*

logging/current_cost]w�;�6Y�+       ��K	���A�,*

logging/current_cost�w�;eӘ�+       ��K	���A�,*

logging/current_costCy�;���3+       ��K	����A�,*

logging/current_cost�w�;�	]�+       ��K	���A�,*

logging/current_cost�w�;/:�(+       ��K	�?��A�,*

logging/current_costbw�;���+       ��K	�l��A�,*

logging/current_cost{�;-��+       ��K	��A�,*

logging/current_costew�;��+       ��K	����A�,*

logging/current_cost�z�;��F+       ��K	����A�,*

logging/current_cost`w�;�+       ��K	�#��A�,*

logging/current_cost8|�;�9+       ��K	�O��A�,*

logging/current_cost�v�;�2�+       ��K	�~��A�,*

logging/current_costIy�;�#Po+       ��K	����A�,*

logging/current_cost w�;۸c�+       ��K	����A�-*

logging/current_cost�y�;$��+       ��K	���A�-*

logging/current_cost�v�;��H+       ��K	�4��A�-*

logging/current_costy�;��+       ��K	ob��A�-*

logging/current_cost.w�;N���+       ��K	���A�-*

logging/current_cost�y�;.]+       ��K	����A�-*

logging/current_cost�v�;�z|4+       ��K	����A�-*

logging/current_cost]x�;�ɩI+       ��K	�&��A�-*

logging/current_cost�w�;p���+       ��K	iU��A�-*

logging/current_costw�;��2+       ��K	����A�-*

logging/current_cost�w�;���+       ��K	����A�-*

logging/current_cost*x�;U�[�+       ��K	\���A�-*

logging/current_cost�w�;���1+       ��K	���A�-*

logging/current_cost�x�;�qͼ+       ��K	-6��A�-*

logging/current_costmw�;��rR+       ��K	<d��A�-*

logging/current_costSy�;|�c�+       ��K	����A�-*

logging/current_cost�w�;ux?+       ��K	����A�-*

logging/current_cost�x�;W�J�+       ��K	����A�-*

logging/current_costw�;<P�a+       ��K	���A�-*

logging/current_cost�x�;�	�i+       ��K	 K��A�-*

logging/current_costw�;���k+       ��K	o}��A�-*

logging/current_costy�;�>�+       ��K	����A�-*

logging/current_cost)w�;�UB�+       ��K	J���A�-*

logging/current_cost�x�;�{hN+       ��K	���A�-*

logging/current_cost�w�;��k;+       ��K	�4��A�-*

logging/current_cost�x�;�Py+       ��K	Zd��A�-*

logging/current_cost1w�;?}-+       ��K	���A�.*

logging/current_costUx�;b~�d+       ��K	=���A�.*

logging/current_cost�w�;���+       ��K	g���A�.*

logging/current_cost�x�;�
��+       ��K	!��A�.*

logging/current_costw�;���+       ��K	;P��A�.*

logging/current_cost�w�;��Dr+       ��K	M��A�.*

logging/current_cost#x�;N��+       ��K	]���A�.*

logging/current_cost�x�;� �+       ��K	����A�.*

logging/current_costFw�;u�a<+       ��K	���A�.*

logging/current_cost�x�;ZYP�+       ��K	;��A�.*

logging/current_cost�w�;F��+       ��K	�m��A�.*

logging/current_cost�w�;��C+       ��K	ݜ��A�.*

logging/current_cost�w�;�|�%+       ��K	=���A�.*

logging/current_costy�;V���+       ��K	����A�.*

logging/current_costpw�;d��+       ��K	�'���A�.*

logging/current_costAx�;:�{�+       ��K	JW���A�.*

logging/current_cost�w�;��n�+       ��K	�����A�.*

logging/current_cost�x�;�X�I+       ��K	�����A�.*

logging/current_cost�w�;�*߷+       ��K	�����A�.*

logging/current_costw�;:�A�+       ��K	���A�.*

logging/current_cost�x�;��V�+       ��K	�C��A�.*

logging/current_cost!w�;-]�}+       ��K	�r��A�.*

logging/current_cost�y�;d�P+       ��K	B���A�.*

logging/current_cost
w�;����+       ��K	q���A�.*

logging/current_cost�w�;B�{�+       ��K	����A�.*

logging/current_cost'w�;X�t+       ��K	\,��A�.*

logging/current_cost9y�;Q�Q3+       ��K	,]��A�/*

logging/current_cost�v�;_�p+       ��K	T���A�/*

logging/current_costzx�;��Vh+       ��K	U���A�/*

logging/current_costw�;�u%+       ��K	����A�/*

logging/current_costy�;n�=�+       ��K	���A�/*

logging/current_cost�v�;��Q�+       ��K	�I��A�/*

logging/current_cost�x�;Nt)+       ��K	�w��A�/*

logging/current_costx�;6ek5+       ��K	@���A�/*

logging/current_cost�w�;fAW�+       ��K	���A�/*

logging/current_cost�x�;6�+       ��K	n���A�/*

logging/current_costy�;�*<+       ��K	�,��A�/*

logging/current_cost�w�;��t+       ��K	uZ��A�/*

logging/current_costx�;a "�+       ��K	���A�/*

logging/current_costx�;�I��+       ��K	����A�/*

logging/current_costDw�;�B7�+       ��K	����A�/*

logging/current_cost�w�;m�/�+       ��K	���A�/*

logging/current_costTw�;,Ԛg+       ��K	[<��A�/*

logging/current_cost:w�;���Y+       ��K	mj��A�/*

logging/current_cost�w�;��ao+       ��K	v���A�/*

logging/current_cost*w�;�KT+       ��K	����A�/*

logging/current_costcw�;�6T+       ��K	����A�/*

logging/current_cost7w�;��)�+       ��K	~��A�/*

logging/current_cost�w�;yD��+       ��K	�J��A�/*

logging/current_cost�w�;���z+       ��K	�w��A�/*

logging/current_costHw�;WK��+       ��K	M���A�/*

logging/current_costmx�;D�Ğ+       ��K	����A�0*

logging/current_costz�;����+       ��K	���A�0*

logging/current_costFw�;�V��+       ��K	.���A�0*

logging/current_cost�v�;�}�_+       ��K	�[���A�0*

logging/current_cost�v�;�0+       ��K	�����A�0*

logging/current_cost0z�;T�x�+       ��K	P����A�0*

logging/current_costdx�;�m�+       ��K	#����A�0*

logging/current_cost�w�;+�އ+       ��K	���A�0*

logging/current_cost�x�;�+�g+       ��K	�@���A�0*

logging/current_costQw�;�.�+       ��K	�l���A�0*

logging/current_cost5x�;E���+       ��K	����A�0*

logging/current_cost�w�;h��+       ��K	����A�0*

logging/current_cost)w�;���_+       ��K	�����A�0*

logging/current_cost1w�;���}+       ��K	� ���A�0*

logging/current_costw�;s�+       ��K	�M���A�0*

logging/current_cost$w�;g��+       ��K	Q~���A�0*

logging/current_cost�v�;ߐ11+       ��K	����A�0*

logging/current_costkx�;◴�+       ��K	�����A�0*

logging/current_cost0w�;/�++       ��K		���A�0*

logging/current_cost�w�;�O�+       ��K	�9���A�0*

logging/current_cost�w�;���b+       ��K	�e���A�0*

logging/current_costw�;�-+       ��K	Β���A�0*

logging/current_costw�;�8�+       ��K	�����A�0*

logging/current_cost�w�;�6,�+       ��K	�����A�0*

logging/current_costVx�;j��+       ��K	w!���A�0*

logging/current_costw�;Q�t+       ��K	�Q���A�0*

logging/current_cost�w�;����+       ��K	z���A�1*

logging/current_cost�v�;TE9�+       ��K	�����A�1*

logging/current_costy�;��+       ��K	����A�1*

logging/current_cost9w�;��R+       ��K	���A�1*

logging/current_cost_x�;�e�k+       ��K	:���A�1*

logging/current_cost�w�;Q+       ��K	7g���A�1*

logging/current_cost�w�;[���+       ��K	 ����A�1*

logging/current_cost�w�;�_�+       ��K	�����A�1*

logging/current_cost�w�;���+       ��K	�����A�1*

logging/current_costvw�;7��+       ��K	����A�1*

logging/current_cost)w�;>�2�+       ��K	.O���A�1*

logging/current_cost�x�;v�F+       ��K	�{���A�1*

logging/current_costaw�;l�K+       ��K	h����A�1*

logging/current_costIw�;����+       ��K	u����A�1*

logging/current_costCw�;�Ю+       ��K	i���A�1*

logging/current_cost�x�;��+       ��K	4���A�1*

logging/current_costmw�;pk��+       ��K	�a���A�1*

logging/current_costYw�;�P[�+       ��K	�����A�1*

logging/current_cost?w�;y�S+       ��K	�����A�1*

logging/current_costfx�;�ձ+       ��K	�@���A�1*

logging/current_cost�w�;X���+       ��K	�����A�1*

logging/current_costTw�;;0�5+       ��K	����A�1*

logging/current_cost.x�;�>��+       ��K	�����A�1*

logging/current_cost1w�;Q0F+       ��K	�%���A�1*

logging/current_cost�w�;�*�+       ��K	�U���A�1*

logging/current_cost�w�;d���+       ��K	i����A�2*

logging/current_costKx�;nR�e+       ��K	�����A�2*

logging/current_cost�w�;�ڢ�+       ��K	!����A�2*

logging/current_cost�x�;�l�+       ��K	m,���A�2*

logging/current_cost(w�;G��+       ��K	d^���A�2*

logging/current_cost`y�;Nd +       ��K	����A�2*

logging/current_cost�x�;�z��+       ��K	�����A�2*

logging/current_cost�w�;�3DS+       ��K	�����A�2*

logging/current_costx�;���+       ��K	����A�2*

logging/current_cost�x�;�3�L+       ��K	zO���A�2*

logging/current_costPw�;��-�+       ��K	�}���A�2*

logging/current_costoz�;WH�+       ��K	M����A�2*

logging/current_cost�v�;��q,+       ��K	�����A�2*

logging/current_cost�w�;��`�+       ��K	 ��A�2*

logging/current_costdw�;9J�9+       ��K	�3 ��A�2*

logging/current_costDx�;��+       ��K	�b ��A�2*

logging/current_cost�z�;%�A�+       ��K	�� ��A�2*

logging/current_cost�v�;�c+       ��K	,� ��A�2*

logging/current_cost�w�;Sp M+       ��K	�� ��A�2*

logging/current_costrw�;"�Vj+       ��K	���A�2*

logging/current_cost�v�;���m+       ��K	�I��A�2*

logging/current_costFw�;J�c�+       ��K	�z��A�2*

logging/current_costWw�;�ϕ+       ��K		���A�2*

logging/current_cost�v�;+k7�+       ��K	���A�2*

logging/current_cost+x�;��C�+       ��K	���A�2*

logging/current_costay�;Ȝؕ+       ��K	h>��A�2*

logging/current_costzy�;�H�+       ��K	�m��A�3*

logging/current_cost5w�;� 4+       ��K	����A�3*

logging/current_cost�w�;����+       ��K	'���A�3*

logging/current_costHw�;hx��+       ��K	:���A�3*

logging/current_cost�x�;L6��+       ��K	%��A�3*

logging/current_cost�w�;�a�+       ��K	+T��A�3*

logging/current_cost�w�;�z+       ��K	���A�3*

logging/current_cost�y�;�g�+       ��K	u���A�3*

logging/current_cost�w�;;O{�+       ��K	����A�3*

logging/current_cost;x�;���+       ��K	���A�3*

logging/current_cost�{�;7�l+       ��K	�C��A�3*

logging/current_cost�w�;����+       ��K	�p��A�3*

logging/current_cost1w�;�R��+       ��K	���A�3*

logging/current_cost�y�;e�7�+       ��K	����A�3*

logging/current_costgw�;V!�=+       ��K	v���A�3*

logging/current_costw�;8�q�+       ��K	�0��A�3*

logging/current_cost�v�;H�+       ��K	=a��A�3*

logging/current_cost�x�;�@?�+       ��K	���A�3*

logging/current_cost!w�;�h�+       ��K	I���A�3*

logging/current_costkw�;�7�+       ��K	@���A�3*

logging/current_costCw�;H��+       ��K	V��A�3*

logging/current_costkx�;,y&!+       ��K	�J��A�3*

logging/current_cost�v�;
0�d+       ��K	rx��A�3*

logging/current_costx�;an+       ��K	̩��A�3*

logging/current_cost�y�;X8�u+       ��K	%���A�3*

logging/current_cost�w�;�1L�+       ��K	���A�3*

logging/current_cost�w�;��+       ��K	�5��A�4*

logging/current_cost�y�;$��+       ��K	�e��A�4*

logging/current_cost�w�;����+       ��K	ѓ��A�4*

logging/current_costw�;�=n�+       ��K	���A�4*

logging/current_costw�;u�R+       ��K	C���A�4*

logging/current_cost�w�;݋�*+       ��K	7��A�4*

logging/current_cost�y�;��7C+       ��K	�M��A�4*

logging/current_cost�v�;�;�l+       ��K	���A�4*

logging/current_cost�w�;�=+       ��K	ͯ��A�4*

logging/current_cost�w�;EYz+       ��K	����A�4*

logging/current_cost�w�; ֚+       ��K	7	��A�4*

logging/current_costz�;��T�+       ��K	�>	��A�4*

logging/current_cost�v�;Ϝ+       ��K	o	��A�4*

logging/current_cost�w�;j�n@+       ��K	�	��A�4*

logging/current_cost�y�;(y_#+       ��K	<�	��A�4*

logging/current_cost�w�;��No+       ��K	��	��A�4*

logging/current_costiw�;b1�+       ��K	�'
��A�4*

logging/current_cost�x�;(r��+       ��K	$W
��A�4*

logging/current_costYw�;"K��+       ��K	��
��A�4*

logging/current_costx�;���+       ��K	�
��A�4*

logging/current_cost�w�;P_�+       ��K	 �
��A�4*

logging/current_costvw�;J�+       ��K	���A�4*

logging/current_cost�w�;ֵ+N+       ��K	�=��A�4*

logging/current_costw�;7Td�+       ��K	�m��A�4*

logging/current_cost�x�;$<�+       ��K	I���A�4*

logging/current_costx�;a��^+       ��K	(���A�5*

logging/current_cost~w�;��tX+       ��K	y���A�5*

logging/current_cost9x�;��;j+       ��K	U$��A�5*

logging/current_costw�;F6r�+       ��K		R��A�5*

logging/current_cost�w�;���+       ��K	���A�5*

logging/current_cost�w�;���+       ��K	����A�5*

logging/current_cost�w�;B4�+       ��K	����A�5*

logging/current_costjw�;{1Y�+       ��K	���A�5*

logging/current_cost
x�;PA�+       ��K	5��A�5*

logging/current_costAw�;C��+       ��K		b��A�5*

logging/current_cost�w�;��)�+       ��K	���A�5*

logging/current_costw�;wc)6+       ��K	H���A�5*

logging/current_cost
x�;��H+       ��K	#���A�5*

logging/current_costHw�;V	%+       ��K	91��A�5*

logging/current_cost�w�;NLu"+       ��K	+y��A�5*

logging/current_cost�x�;����+       ��K	[���A�5*

logging/current_cost�w�;j�n+       ��K	^���A�5*

logging/current_cost�w�;>G +       ��K	�)��A�5*

logging/current_cost�x�;޼�+       ��K	�p��A�5*

logging/current_costYw�;�b|+       ��K	����A�5*

logging/current_costgy�;S���+       ��K	���A�5*

logging/current_cost�v�;Ó+       ��K	#)��A�5*

logging/current_cost�w�;�I &+       ��K	�[��A�5*

logging/current_costjw�;����+       ��K	Ӗ��A�5*

logging/current_cost�w�;$��+       ��K	����A�5*

logging/current_cost�w�;��O9+       ��K	���A�5*

logging/current_cost�v�;$)}T+       ��K	�?��A�6*

logging/current_cost�w�;�R�+       ��K	wt��A�6*

logging/current_cost�w�;�)��+       ��K	Ŧ��A�6*

logging/current_cost{w�;M|�+       ��K	���A�6*

logging/current_costw�;�P_�+       ��K	�#��A�6*

logging/current_costy�;��w[+       ��K	�_��A�6*

logging/current_costuw�;��*�+       ��K	ܐ��A�6*

logging/current_cost�w�;%�Ik+       ��K	P���A�6*

logging/current_cost�x�;ٯ�3+       ��K	.���A�6*

logging/current_cost�x�;�"SK+       ��K	G%��A�6*

logging/current_costJw�;���+       ��K	0S��A�6*

logging/current_cost�w�;�_��+       ��K	I���A�6*

logging/current_cost�w�;��E�+       ��K	����A�6*

logging/current_cost_w�;��L�+       ��K	���A�6*

logging/current_cost�w�;��y�+       ��K	L��A�6*

logging/current_cost=w�;��w+       ��K	BG��A�6*

logging/current_cost�v�;�d <+       ��K	�w��A�6*

logging/current_cost�w�;�Om�+       ��K	���A�6*

logging/current_cost�x�;��+       ��K	@���A�6*

logging/current_costNw�;.�p+       ��K	�!��A�6*

logging/current_cost[x�;��Am+       ��K	�N��A�6*

logging/current_cost�w�;��>+       ��K	:z��A�6*

logging/current_cost)w�;J�m5+       ��K	����A�6*

logging/current_cost�v�;�a+       ��K	P���A�6*

logging/current_cost�w�;��=�+       ��K	��A�6*

logging/current_cost$w�;����+       ��K	�A��A�7*

logging/current_cost�w�;�w��+       ��K	Dl��A�7*

logging/current_cost�w�;�	�+       ��K	כ��A�7*

logging/current_costw�;�H4�+       ��K	����A�7*

logging/current_costVw�;`�+       ��K		���A�7*

logging/current_cost;w�;�d�+       ��K	L*��A�7*

logging/current_cost4x�;K���+       ��K	�W��A�7*

logging/current_cost6w�;��=+       ��K	���A�7*

logging/current_cost�v�;�ϻ�+       ��K	J���A�7*

logging/current_cost�v�;���g+       ��K	���A�7*

logging/current_cost�w�;�Fl�+       ��K	��A�7*

logging/current_cost�w�;-���+       ��K	�<��A�7*

logging/current_cost�v�;wp�+       ��K	6i��A�7*

logging/current_costpx�;���+       ��K	���A�7*

logging/current_costnw�;w��+       ��K	\���A�7*

logging/current_cost>w�;ќ�3+       ��K	����A�7*

logging/current_cost�x�;� �F+       ��K	� ��A�7*

logging/current_cost+w�;���+       ��K	�P��A�7*

logging/current_cost�w�;��פ+       ��K	�~��A�7*

logging/current_cost:w�;� �+       ��K	T���A�7*

logging/current_cost�y�;�1��+       ��K	.���A�7*

logging/current_costw�;o^>+       ��K	W��A�7*

logging/current_cost1x�;�i	+       ��K	�7��A�7*

logging/current_costRw�;�M8�+       ��K	g��A�7*

logging/current_cost^w�;���+       ��K	����A�7*

logging/current_cost�w�;n���+       ��K	����A�7*

logging/current_costmw�;T�5+       ��K	i���A�8*

logging/current_costbw�;���+       ��K	<1��A�8*

logging/current_costw�;Ma�+       ��K	�`��A�8*

logging/current_cost�x�;�M��+       ��K	����A�8*

logging/current_costlw�;p�X+       ��K	����A�8*

logging/current_costMw�;3jȥ+       ��K	���A�8*

logging/current_costw�;��=U+       ��K	���A�8*

logging/current_cost�x�;̱O�+       ��K	�H��A�8*

logging/current_cost!w�;ѳG�+       ��K	[x��A�8*

logging/current_cost�y�;�{��+       ��K	����A�8*

logging/current_cost�x�;�]�+       ��K	����A�8*

logging/current_costw�;�c�)+       ��K	g��A�8*

logging/current_cost,x�;��+       ��K	�4��A�8*

logging/current_cost�x�;��Ϡ+       ��K	~c��A�8*

logging/current_cost.w�;�ʀ8+       ��K	|���A�8*

logging/current_costy�;��^�+       ��K	����A�8*

logging/current_costQy�;�A2+       ��K	C���A�8*

logging/current_cost�v�;��+       ��K	�,��A�8*

logging/current_cost�w�;��+       ��K	�[��A�8*

logging/current_cost2w�;��9�+       ��K	����A�8*

logging/current_costrw�;����+       ��K	!���A�8*

logging/current_cost!w�;9֞+       ��K	���A�8*

logging/current_cost�v�;���+       ��K	���A�8*

logging/current_cost�w�;i��+       ��K	F��A�8*

logging/current_costw�;@��+       ��K	�v��A�8*

logging/current_cost�w�;ya�2+       ��K	����A�8*

logging/current_cost w�;+�+       ��K	����A�9*

logging/current_cost�x�;*�-6+       ��K	� ��A�9*

logging/current_cost2w�;���+       ��K	�5 ��A�9*

logging/current_costFw�;����+       ��K	ki ��A�9*

logging/current_cost�w�;4r�!+       ��K	Θ ��A�9*

logging/current_costtw�;��d+       ��K	�� ��A�9*

logging/current_cost`w�;뻵�+       ��K	I� ��A�9*

logging/current_cost)w�;x��+       ��K	�&!��A�9*

logging/current_costMx�;�Z��+       ��K	3T!��A�9*

logging/current_cost-w�;�r�+       ��K	^�!��A�9*

logging/current_cost9w�;6�8�+       ��K	\�!��A�9*

logging/current_cost�w�;���+       ��K	��!��A�9*

logging/current_cost�w�;��s3+       ��K	�"��A�9*

logging/current_cost�v�;Erz+       ��K	x?"��A�9*

logging/current_costy�;���Q+       ��K	�l"��A�9*

logging/current_cost*w�;K��+       ��K	��"��A�9*

logging/current_cost�w�;"K�+       ��K	k�"��A�9*

logging/current_cost�w�;��t2+       ��K	�"��A�9*

logging/current_cost�v�;����+       ��K	�&#��A�9*

logging/current_costky�;��i+       ��K	�S#��A�9*

logging/current_costw�;�d�T+       ��K	��#��A�9*

logging/current_costow�;��2.+       ��K	&�#��A�9*

logging/current_cost�w�;���+       ��K	�#��A�9*

logging/current_costx�;K=�+       ��K	�$��A�9*

logging/current_cost�w�;�+�+       ��K	�;$��A�9*

logging/current_costy�;�9�t+       ��K	h$��A�:*

logging/current_costw�;Cl6M+       ��K	��$��A�:*

logging/current_cost�w�;��q�+       ��K	L�$��A�:*

logging/current_costiw�;���+       ��K	��$��A�:*

logging/current_cost�x�;��=4+       ��K	� %��A�:*

logging/current_cost`w�;���+       ��K	�P%��A�:*

logging/current_cost�v�;��T�+       ��K	a~%��A�:*

logging/current_costx�;��-�+       ��K	R�%��A�:*

logging/current_cost�w�;Lh{�+       ��K	��%��A�:*

logging/current_costCw�;gt/5+       ��K	V
&��A�:*

logging/current_costSw�;¢��+       ��K	�7&��A�:*

logging/current_costzx�;�ޝK+       ��K	�e&��A�:*

logging/current_cost�w�;1g6�+       ��K	˕&��A�:*

logging/current_cost�w�;մi�+       ��K	��&��A�:*

logging/current_costw�;ٿ6Q+       ��K	+�&��A�:*

logging/current_costgw�;�.�'+       ��K	D('��A�:*

logging/current_cost8w�;)�f+       ��K	�U'��A�:*

logging/current_cost�x�;��+       ��K	��'��A�:*

logging/current_cost w�;���+       ��K	�'��A�:*

logging/current_cost(x�;��׳+       ��K	��'��A�:*

logging/current_costhx�;�ip+       ��K	�(��A�:*

logging/current_cost�w�;C;�+       ��K	m<(��A�:*

logging/current_costz�;���A+       ��K	�k(��A�:*

logging/current_costtw�;��*R+       ��K	Ř(��A�:*

logging/current_costyw�; �Z:+       ��K	6�(��A�:*

logging/current_cost�w�;l���+       ��K	��(��A�:*

logging/current_costzw�;Ţ�g+       ��K	� )��A�;*

logging/current_cost�x�;��D>+       ��K	N)��A�;*

logging/current_cost\w�;N�z+       ��K	!})��A�;*

logging/current_cost�v�;���+       ��K	�)��A�;*

logging/current_costx�;���D+       ��K	c�)��A�;*

logging/current_cost�w�;�f��+       ��K	N*��A�;*

logging/current_cost]w�;��e�+       ��K	�4*��A�;*

logging/current_costAx�;���x+       ��K	Rb*��A�;*

logging/current_cost�w�;g-�;+       ��K	��*��A�;*

logging/current_cost�w�;*ͷq+       ��K	�*��A�;*

logging/current_cost�w�;b!3[+       ��K	��*��A�;*

logging/current_costFw�;�'�+       ��K	�%+��A�;*

logging/current_cost�x�;��p+       ��K	�U+��A�;*

logging/current_cost�w�;&��=+       ��K	�+��A�;*

logging/current_cost�w�;�u��+       ��K	��+��A�;*

logging/current_costw�;W_�+       ��K	j�+��A�;*

logging/current_costy�;�o(�+       ��K	�,��A�;*

logging/current_costCw�;��w�+       ��K	�;,��A�;*

logging/current_cost(x�;��B�+       ��K	�m,��A�;*

logging/current_costyy�;�p�+       ��K	��,��A�;*

logging/current_cost{y�;^Y�+       ��K	��,��A�;*

logging/current_cost0w�;T�+       ��K	>�,��A�;*

logging/current_costx�;a��+       ��K	C,-��A�;*

logging/current_costw�;�Ϳg+       ��K	�]-��A�;*

logging/current_cost�w�;ǙV�+       ��K	��-��A�;*

logging/current_cost#w�;���+       ��K	'�-��A�<*

logging/current_cost�x�;zX+       ��K	��-��A�<*

logging/current_cost�v�;M���+       ��K	U.��A�<*

logging/current_cost
x�;9��+       ��K	�E.��A�<*

logging/current_costaw�;�˗�+       ��K	As.��A�<*

logging/current_costx�;�k��+       ��K	m�.��A�<*

logging/current_cost)z�;��X�+       ��K	%�.��A�<*

logging/current_costaw�;�N�+       ��K	� /��A�<*

logging/current_cost�x�;1�+       ��K	�,/��A�<*

logging/current_cost�w�;Q�`h+       ��K	�\/��A�<*

logging/current_cost[w�;ƿ
+       ��K	��/��A�<*

logging/current_cost�y�;�lڀ+       ��K	�/��A�<*

logging/current_costx�;��+       ��K	��/��A�<*

logging/current_costw�;mi$Z+       ��K	�0��A�<*

logging/current_costx�;�x�+       ��K	�F0��A�<*

logging/current_costDx�;>�+       ��K	Gu0��A�<*

logging/current_cost�w�;�UV+       ��K	��0��A�<*

logging/current_costx�;��V�+       ��K	��0��A�<*

logging/current_cost
x�;��r+       ��K	� 1��A�<*

logging/current_cost�w�;���+       ��K	�.1��A�<*

logging/current_cost�w�;�d�+       ��K	@\1��A�<*

logging/current_costgw�;R��+       ��K	5�1��A�<*

logging/current_cost�w�;-]`�+       ��K	��1��A�<*

logging/current_cost�w�;�f�l+       ��K	��1��A�<*

logging/current_costIw�;*�$+       ��K	q2��A�<*

logging/current_cost�w�;#���+       ��K	�C2��A�<*

logging/current_cost�w�;�Zn�+       ��K	�p2��A�=*

logging/current_cost�w�;��Ow+       ��K	Ŝ2��A�=*

logging/current_costew�;;�wG+       ��K	��2��A�=*

logging/current_costdw�;�4V+       ��K	`�2��A�=*

logging/current_cost�w�;�Sa+       ��K	V"3��A�=*

logging/current_cost�w�;u�h}+       ��K	�P3��A�=*

logging/current_costx�;��b)+       ��K	�~3��A�=*

logging/current_cost<w�;��B+       ��K	��3��A�=*

logging/current_cost�x�;�ԩ�+       ��K	d�3��A�=*

logging/current_cost2w�;dR�+       ��K	O4��A�=*

logging/current_cost�w�;ߋk�+       ��K	j44��A�=*

logging/current_costtw�;m���+       ��K	a4��A�=*

logging/current_cost�x�;c��+       ��K	��4��A�=*

logging/current_costEw�;���j+       ��K	�4��A�=*

logging/current_cost�w�;�.�a+       ��K	�4��A�=*

logging/current_cost}w�;��5�+       ��K	�5��A�=*

logging/current_cost�x�;@s&y+       ��K	MD5��A�=*

logging/current_cost<w�;�ڀw+       ��K	Vr5��A�=*

logging/current_costw�;K@+       ��K	_�5��A�=*

logging/current_cost$x�;��ȫ+       ��K	��5��A�=*

logging/current_cost�w�;ĒL|+       ��K	��5��A�=*

logging/current_costw�;!� +       ��K	f%6��A�=*

logging/current_cost-w�;��]�+       ��K	�V6��A�=*

logging/current_costWx�; �3+       ��K	:�6��A�=*

logging/current_cost�w�;�+       ��K	+�6��A�=*

logging/current_costJw�;2�]�+       ��K	��6��A�=*

logging/current_cost�w�;ü`=+       ��K	G7��A�>*

logging/current_cost{w�;Z�q�+       ��K	�97��A�>*

logging/current_cost&x�;�ܞ�+       ��K	�d7��A�>*

logging/current_costyw�;�,�+       ��K	l�7��A�>*

logging/current_costCw�;�U*+       ��K	]�7��A�>*

logging/current_cost[x�;Y�tA+       ��K	/�7��A�>*

logging/current_cost�w�;tQ�Z+       ��K	8��A�>*

logging/current_cost�w�;Z\��+       ��K	�N8��A�>*

logging/current_costw�;Շ�'+       ��K	R{8��A�>*

logging/current_costjw�;��+       ��K	U�8��A�>*

logging/current_cost:w�;���+       ��K	y�8��A�>*

logging/current_costPw�;+7��+       ��K	�9��A�>*

logging/current_cost�v�;9�*�+       ��K	669��A�>*

logging/current_costw�;W�4+       ��K	�c9��A�>*

logging/current_cost�v�;0��
+       ��K	��9��A�>*

logging/current_costy�;�^�+       ��K	[�9��A�>*

logging/current_cost�v�;��)+       ��K	o�9��A�>*

logging/current_costVw�;�c,+       ��K	]":��A�>*

logging/current_costZw�;�v �+       ��K	 V:��A�>*

logging/current_cost1w�;�|+       ��K	چ:��A�>*

logging/current_cost�w�;���+       ��K	��:��A�>*

logging/current_cost~w�;��-�+       ��K	E�:��A�>*

logging/current_cost[w�;�k��+       ��K	�;��A�>*

logging/current_cost�w�;��o4+       ��K	<;��A�>*

logging/current_costw�;Oz�+       ��K	�v;��A�>*

logging/current_cost%w�;+��+       ��K	'�;��A�?*

logging/current_costw�;�A�+       ��K	�<��A�?*

logging/current_cost�w�;���+       ��K	6M<��A�?*

logging/current_cost�v�;=z�+       ��K	̤<��A�?*

logging/current_cost�v�;�D1+       ��K	X�<��A�?*

logging/current_costw�;]2u4+       ��K	=��A�?*

logging/current_cost�w�;r@?�+       ��K	GF=��A�?*

logging/current_costw�;i~��+       ��K	�=��A�?*

logging/current_cost�v�;�H��+       ��K	��=��A�?*

logging/current_cost~x�;Wz�+       ��K	�>��A�?*

logging/current_cost�w�;rI˿+       ��K	�;>��A�?*

logging/current_costPw�;T�X�+       ��K	*t>��A�?*

logging/current_cost9w�;
��k+       ��K	i�>��A�?*

logging/current_costQw�;��S+       ��K	��>��A�?*

logging/current_costx�;�br+       ��K	�?��A�?*

logging/current_costsw�;���+       ��K	r1?��A�?*

logging/current_cost�v�;��<+       ��K	*g?��A�?*

logging/current_costXw�;U�ՠ+       ��K	��?��A�?*

logging/current_cost�x�;G��u+       ��K	�?��A�?*

logging/current_cost�v�;x�+       ��K	��?��A�?*

logging/current_cost�x�;�%k�+       ��K	� @��A�?*

logging/current_cost.w�;�Mb�+       ��K	�M@��A�?*

logging/current_cost�y�;q��+       ��K	�|@��A�?*

logging/current_costx�;�5+       ��K	H�@��A�?*

logging/current_costew�;��c+       ��K	9�@��A�?*

logging/current_cost�w�;��Cq+       ��K	kA��A�?*

logging/current_costx�;�3B�+       ��K	�6A��A�@*

logging/current_cost�w�;�7�>+       ��K	�dA��A�@*

logging/current_costcw�;�v�+       ��K	��A��A�@*

logging/current_costgz�;��V+       ��K	U�A��A�@*

logging/current_cost�w�;�%1�+       ��K	D�A��A�@*

logging/current_cost�w�;��P�+       ��K	mB��A�@*

logging/current_cost�v�;K{5�+       ��K	�LB��A�@*

logging/current_cost�x�;��+       ��K	�B��A�@*

logging/current_cost�w�;�㪅+       ��K	��B��A�@*

logging/current_cost�w�;m���+       ��K	��B��A�@*

logging/current_costw�;}�:�+       ��K	�C��A�@*

logging/current_cost2y�;��0�+       ��K	3C��A�@*

logging/current_cost~w�;�9ct+       ��K	*aC��A�@*

logging/current_costbw�;%��+       ��K	S�C��A�@*

logging/current_costw�;&lwW+       ��K	K�C��A�@*

logging/current_cost�x�;�Y:+       ��K	>�C��A�@*

logging/current_cost�w�;���+       ��K	�D��A�@*

logging/current_cost�w�;~.ְ+       ��K	8CD��A�@*

logging/current_cost,w�;y<� +       ��K	}pD��A�@*

logging/current_cost�x�;���9+       ��K	R�D��A�@*

logging/current_cost`w�;Q�-+       ��K	��D��A�@*

logging/current_costjw�;�g	+       ��K	y�D��A�@*

logging/current_cost�w�;�[Y�+       ��K	�+E��A�@*

logging/current_cost�w�;��(�+       ��K	�\E��A�@*

logging/current_cost�v�;��+       ��K	��E��A�@*

logging/current_cost�y�;�=��+       ��K	M�E��A�A*

logging/current_cost^w�;9�+       ��K	n�E��A�A*

logging/current_cost�w�;��+       ��K	�F��A�A*

logging/current_cost*w�;y�K+       ��K	�FF��A�A*

logging/current_costw�;�0g�+       ��K	xuF��A�A*

logging/current_costfw�;�rY�+       ��K	��F��A�A*

logging/current_cost�w�;.�M+       ��K	��F��A�A*

logging/current_cost)x�;5�+       ��K	�G��A�A*

logging/current_costzx�;�� M+       ��K	�1G��A�A*

logging/current_cost�v�;Mi\�+       ��K	�`G��A�A*

logging/current_cost�w�;���+       ��K	��G��A�A*

logging/current_cost�v�;x�87+       ��K	��G��A�A*

logging/current_costz�;@�K+       ��K	�H��A�A*

logging/current_cost�v�;Ka� +       ��K	UUH��A�A*

logging/current_cost�x�;[Ю+       ��K	��H��A�A*

logging/current_costw�;�<�+       ��K	��H��A�A*

logging/current_cost6y�;�#2+       ��K	�
I��A�A*

logging/current_cost�v�;&���+       ��K	7FI��A�A*

logging/current_cost8x�;�ϵ+       ��K	�I��A�A*

logging/current_costw�;�)ߪ+       ��K	T�I��A�A*

logging/current_costry�;�4+       ��K	XJ��A�A*

logging/current_cost�v�;Q�+       ��K	�OJ��A�A*

logging/current_costjx�;�&a7+       ��K	l�J��A�A*

logging/current_cost1w�;n��+       ��K	p�J��A�A*

logging/current_cost'w�;��E�+       ��K	�J��A�A*

logging/current_costw�;��B++       ��K	r:K��A�A*

logging/current_cost_w�;8v�+       ��K	kK��A�B*

logging/current_cost+w�;Тu+       ��K	K�K��A�B*

logging/current_cost�w�;�s��+       ��K	,�K��A�B*

logging/current_cost�w�;��+       ��K	��K��A�B*

logging/current_costQw�;J�a?+       ��K	�+L��A�B*

logging/current_costMw�;���+       ��K	�_L��A�B*

logging/current_costw�;܏�+       ��K	 �L��A�B*

logging/current_cost7w�;K��+       ��K	��L��A�B*

logging/current_costew�;� +       ��K	��L��A�B*

logging/current_cost�w�;�W0Z+       ��K	�M��A�B*

logging/current_cost�w�;�=+�+       ��K	�IM��A�B*

logging/current_cost�w�;.,f+       ��K	 yM��A�B*

logging/current_cost�v�;�B�+       ��K	�M��A�B*

logging/current_cost�x�;��=H+       ��K	�M��A�B*

logging/current_cost�v�;�:+       ��K	�N��A�B*

logging/current_cost5w�;Z-܊+       ��K	�7N��A�B*

logging/current_costDx�;�.��+       ��K	�qN��A�B*

logging/current_cost�w�;����+       ��K	��N��A�B*

logging/current_cost�v�;�>a-+       ��K	��N��A�B*

logging/current_cost�y�;��x+       ��K	��N��A�B*

logging/current_cost�v�;��Cw+       ��K	<+O��A�B*

logging/current_costCy�;|�:a+       ��K	pYO��A�B*

logging/current_cost�w�;�n�+       ��K	��O��A�B*

logging/current_cost�w�;�&|@+       ��K	��O��A�B*

logging/current_costx�;�>��+       ��K	>�O��A�B*

logging/current_cost�w�;pp��+       ��K	�P��A�B*

logging/current_cost�x�;�1�+       ��K	=P��A�C*

logging/current_cost�w�;x-�}+       ��K	+qP��A�C*

logging/current_cost�w�;[� +       ��K	�P��A�C*

logging/current_costmw�;��O�+       ��K	��P��A�C*

logging/current_cost�w�;�w�l+       ��K	��P��A�C*

logging/current_costGw�;#�;�+       ��K	�+Q��A�C*

logging/current_cost�w�;;9V�+       ��K	�[Q��A�C*

logging/current_cost�w�;p<��+       ��K	s�Q��A�C*

logging/current_cost�w�;�M+       ��K	�Q��A�C*

logging/current_costhw�;�͛S+       ��K	��Q��A�C*

logging/current_cost/x�;��m+       ��K	�#R��A�C*

logging/current_costx�;rj4+       ��K	AXR��A�C*

logging/current_cost�w�;����+       ��K	�R��A�C*

logging/current_cost�w�;-
��+       ��K	�R��A�C*

logging/current_cost5w�;��G�+       ��K	�R��A�C*

logging/current_cost]w�;f���+       ��K	@S��A�C*

logging/current_cost�w�;j�(w+       ��K	-BS��A�C*

logging/current_cost x�;2TB�+       ��K	XpS��A�C*

logging/current_cost	w�;Z:+       ��K	�S��A�C*

logging/current_costw�;-D��+       ��K	��S��A�C*

logging/current_cost*x�;;���+       ��K	�T��A�C*

logging/current_costQw�;z���+       ��K	�;T��A�C*

logging/current_cost�x�;�C
4+       ��K	�jT��A�C*

logging/current_cost�w�;�&��+       ��K	��T��A�C*

logging/current_cost�w�;5%+       ��K	t�T��A�C*

logging/current_cost�w�;z� +       ��K	�U��A�D*

logging/current_cost�w�;���+       ��K	�<U��A�D*

logging/current_cost+x�;4��*+       ��K	_kU��A�D*

logging/current_costPw�;�9�)+       ��K	f�U��A�D*

logging/current_cost�w�;��V�+       ��K	P�U��A�D*

logging/current_costWw�;��O+       ��K	B�U��A�D*

logging/current_cost�x�;��+       ��K	�,V��A�D*

logging/current_cost�w�;6.�;+       ��K	O[V��A�D*

logging/current_cost)w�;_�+       ��K	؉V��A�D*

logging/current_cost|w�;v�ij+       ��K	зV��A�D*

logging/current_costGy�;v�+       ��K	��V��A�D*

logging/current_costw�;�+��+       ��K	�W��A�D*

logging/current_cost1y�;OV:+       ��K	-FW��A�D*

logging/current_cost�w�;��W\+       ��K	%sW��A�D*

logging/current_costxw�;Km�"+       ��K	��W��A�D*

logging/current_costz�;��}�+       ��K	��W��A�D*

logging/current_costw�;���I+       ��K	��W��A�D*

logging/current_costAx�;X�6
+       ��K	6/X��A�D*

logging/current_cost�w�;�b�)+       ��K	�[X��A�D*

logging/current_costw�;�Yh+       ��K	�X��A�D*

logging/current_cost�w�;F���+       ��K	̵X��A�D*

logging/current_cost�x�;�꼰+       ��K	t�X��A�D*

logging/current_costw�;�A�+       ��K	�Y��A�D*

logging/current_cost�x�;�m6+       ��K	LJY��A�D*

logging/current_costpw�;���0+       ��K	�xY��A�D*

logging/current_costbw�;��=+       ��K	8�Y��A�D*

logging/current_cost�y�;�^�F+       ��K	��Y��A�E*

logging/current_costw�;�.��+       ��K	�Z��A�E*

logging/current_costx�;���+       ��K	�0Z��A�E*

logging/current_cost�w�;^�!,+       ��K	�aZ��A�E*

logging/current_costw�;�n�+       ��K	��Z��A�E*

logging/current_costmw�;eG�S+       ��K	��Z��A�E*

logging/current_cost_x�;ס�+       ��K	v�Z��A�E*

logging/current_costw�;���+       ��K	0[��A�E*

logging/current_costlx�;���v+       ��K	kF[��A�E*

logging/current_costWw�;��%+       ��K	Mt[��A�E*

logging/current_costXw�;PN+       ��K	��[��A�E*

logging/current_cost�y�;L�S+       ��K	�[��A�E*

logging/current_costw�;^��+       ��K	��[��A�E*

logging/current_costx�;7��!+       ��K	,\��A�E*

logging/current_cost�w�;f�<�+       ��K	�X\��A�E*

logging/current_cost
w�;l$>E+       ��K	Ȇ\��A�E*

logging/current_cost`w�;&&E+       ��K	{�\��A�E*

logging/current_costNx�;=�j+       ��K	t�\��A�E*

logging/current_costw�;�`M+       ��K	p]��A�E*

logging/current_costVx�;и+       ��K	�<]��A�E*

logging/current_costKw�;�+       ��K	!j]��A�E*

logging/current_costRw�;��+       ��K	��]��A�E*

logging/current_costyy�;^�1�+       ��K	��]��A�E*

logging/current_cost�v�;'\+       ��K	��]��A�E*

logging/current_costx�;���+       ��K	�&^��A�E*

logging/current_cost�w�;5��+       ��K	 U^��A�F*

logging/current_costw�;�z��+       ��K	H�^��A�F*

logging/current_costYw�;L�N�+       ��K	��^��A�F*

logging/current_costEx�;8��+       ��K	��^��A�F*

logging/current_costw�;��r�+       ��K	�_��A�F*

logging/current_costHx�;}H��+       ��K	�D_��A�F*

logging/current_costEw�;^俪+       ��K	�r_��A�F*

logging/current_costOw�;T��+       ��K	ޢ_��A�F*

logging/current_costhy�;ҧuJ+       ��K	`�_��A�F*

logging/current_cost�v�;�9u�+       ��K	)�_��A�F*

logging/current_cost�w�;>\+       ��K	)+`��A�F*

logging/current_costw�;���+       ��K	�X`��A�F*

logging/current_costw�;�ڠ�+       ��K	�`��A�F*

logging/current_costUw�;�ܷ�+       ��K	�`��A�F*

logging/current_cost@x�;\-g�+       ��K	p�`��A�F*

logging/current_costw�;;�5j+       ��K	ba��A�F*

logging/current_cost@x�;�i�+       ��K	hGa��A�F*

logging/current_costBw�;�!�g+       ��K	tva��A�F*

logging/current_costMw�;����+       ��K	�a��A�F*

logging/current_cost_y�;��+       ��K	[�a��A�F*

logging/current_cost�v�;�*b+       ��K	�b��A�F*

logging/current_cost�w�;}��+       ��K	o3b��A�F*

logging/current_cost}w�;�[Y+       ��K	�ab��A�F*

logging/current_costw�;�l�+       ��K	�b��A�F*

logging/current_costTw�; |��+       ��K	o�b��A�F*

logging/current_cost=x�;�[^+       ��K	��b��A�F*

logging/current_costw�;�k+       ��K	!c��A�G*

logging/current_cost;x�;1*�+       ��K	-Mc��A�G*

logging/current_costAw�;�P'�+       ��K	�c��A�G*

logging/current_costJw�;�Y+       ��K	��c��A�G*

logging/current_cost[y�;]�J+       ��K	9�c��A�G*

logging/current_cost�v�;98Iy+       ��K	�d��A�G*

logging/current_cost�w�;3d��+       ��K	7d��A�G*

logging/current_cost|w�;����+       ��K	�dd��A�G*

logging/current_cost
w�;��+       ��K	�d��A�G*

logging/current_costRw�;=��
+       ��K	�d��A�G*

logging/current_cost;x�;��<�+       ��K	��d��A�G*

logging/current_costw�;#��+       ��K	2!e��A�G*

logging/current_cost6x�;��+       ��K	RQe��A�G*

logging/current_costAw�;�{��+       ��K	�~e��A�G*

logging/current_costJw�;Ԩ+       ��K	��e��A�G*

logging/current_costXy�;��h+       ��K	U�e��A�G*

logging/current_cost�v�;<.l+       ��K	Cf��A�G*

logging/current_cost�w�;���+       ��K	�3f��A�G*

logging/current_costzw�;cp�V+       ��K	�af��A�G*

logging/current_costw�;����+       ��K	ˑf��A�G*

logging/current_costQw�;/��j+       ��K	��f��A�G*

logging/current_cost;x�;�w�3+       ��K	h�f��A�G*

logging/current_costw�;��AG+       ��K	p-g��A�G*

logging/current_cost7x�;p�T+       ��K	|]g��A�G*

logging/current_cost@w�;�Yv+       ��K	��g��A�G*

logging/current_costIw�;տ"�+       ��K	&�g��A�G*

logging/current_costUy�;��+       ��K	X�g��A�H*

logging/current_cost�v�;A��+       ��K	qh��A�H*

logging/current_cost�w�;4=b�+       ��K	�Eh��A�H*

logging/current_cost{w�;���[+       ��K	Bsh��A�H*

logging/current_costw�;mMQ�+       ��K	ʠh��A�H*

logging/current_costOw�;2�K+       ��K	��h��A�H*

logging/current_cost;x�;�T\+       ��K	�h��A�H*

logging/current_costw�;N�{>+       ��K	�)i��A�H*

logging/current_cost5x�;��֏+       ��K	2Xi��A�H*

logging/current_costAw�;�q��+       ��K	L�i��A�H*

logging/current_costIw�;4�[�+       ��K	9�i��A�H*

logging/current_costSy�;��Vm+       ��K	`�i��A�H*

logging/current_cost�v�;���+       ��K	�
j��A�H*

logging/current_cost�w�;��X+       ��K	T8j��A�H*

logging/current_cost{w�;�{�+       ��K	7gj��A�H*

logging/current_costw�;�.Ź+       ��K	��j��A�H*

logging/current_costPw�;#�+       ��K	��j��A�H*

logging/current_cost;x�;H��+       ��K	<�j��A�H*

logging/current_costw�;J�+       ��K	[k��A�H*

logging/current_cost4x�;&m�+       ��K	`Kk��A�H*

logging/current_costAw�;��a�+       ��K	zk��A�H*

logging/current_costHw�;�I�+       ��K	Ʃk��A�H*

logging/current_costTy�;��<+       ��K	b�k��A�H*

logging/current_cost�v�;�1++       ��K	�	l��A�H*

logging/current_cost�w�;l�;+       ��K	�9l��A�H*

logging/current_costzw�;ۡ�+       ��K	�fl��A�I*

logging/current_costw�;x���+       ��K	K�l��A�I*

logging/current_costPw�;��
+       ��K	/�l��A�I*

logging/current_cost;x�;���i+       ��K	�l��A�I*

logging/current_costw�;����+       ��K	Hm��A�I*

logging/current_cost3x�;2�R�+       ��K	�Hm��A�I*

logging/current_costBw�;��k�+       ��K	iwm��A�I*

logging/current_costJw�;�$c�+       ��K	E�m��A�I*

logging/current_costSy�;��D+       ��K	L�m��A�I*

logging/current_cost�v�;�}�+       ��K	�n��A�I*

logging/current_cost�w�;�3��+       ��K	�3n��A�I*

logging/current_costzw�;�K��+       ��K	lcn��A�I*

logging/current_costw�;���+       ��K	��n��A�I*

logging/current_costPw�;����+       ��K	��n��A�I*

logging/current_cost;x�;�J�+       ��K	v�n��A�I*

logging/current_costw�;��L+       ��K	fo��A�I*

logging/current_cost3x�;�ꁀ+       ��K	�Ko��A�I*

logging/current_costDw�;����+       ��K	yo��A�I*

logging/current_costGw�;�I�>+       ��K	(�o��A�I*

logging/current_costRy�;e���+       ��K	��o��A�I*

logging/current_cost�v�;��+       ��K	�p��A�I*

logging/current_cost�w�;�D�i+       ��K	V2p��A�I*

logging/current_cost{w�;p
R+       ��K	sbp��A�I*

logging/current_cost
w�;�(��+       ��K	��p��A�I*

logging/current_costPw�;�7�X+       ��K	��p��A�I*

logging/current_cost=x�;	��+       ��K	��p��A�I*

logging/current_costw�;X�2Y+       ��K	,q��A�J*

logging/current_cost4x�;е;+       ��K	 Kq��A�J*

logging/current_costDw�;f)|1+       ��K	yq��A�J*

logging/current_costGw�;��"+       ��K	��q��A�J*

logging/current_costQy�;��G+       ��K	{�q��A�J*

logging/current_cost�v�;�ޤ+       ��K	r��A�J*

logging/current_cost�w�;7�r+       ��K	&7r��A�J*

logging/current_cost{w�;L�Jm+       ��K	Ler��A�J*

logging/current_costw�;(���+       ��K	r��A�J*

logging/current_costPw�;�\%+       ��K	��r��A�J*

logging/current_cost=x�;�/o+       ��K	V�r��A�J*

logging/current_costw�;on7�+       ��K	�s��A�J*

logging/current_cost2x�;5�A�+       ��K	Ks��A�J*

logging/current_costFw�;$�L�+       ��K	xs��A�J*

logging/current_costHw�;���1+       ��K	 �s��A�J*

logging/current_costRy�;rn�+       ��K	L�s��A�J*

logging/current_cost�v�;��:�+       ��K	��s��A�J*

logging/current_cost�w�;uoq�+       ��K	2.t��A�J*

logging/current_cost}w�;���
+       ��K	|\t��A�J*

logging/current_cost
w�;���+       ��K	݊t��A�J*

logging/current_costQw�;����+       ��K	'�t��A�J*

logging/current_cost=x�;��64+       ��K	��t��A�J*

logging/current_costw�;=��g+       ��K	'u��A�J*

logging/current_cost3x�;��m+       ��K	kCu��A�J*

logging/current_costHw�;�L+       ��K	1ru��A�J*

logging/current_costFw�;����+       ��K	G�u��A�K*

logging/current_costRy�;����+       ��K	�u��A�K*

logging/current_cost�v�;���Z+       ��K	��u��A�K*

logging/current_cost�w�;�j�+       ��K	T,v��A�K*

logging/current_cost�w�;�MZ5+       ��K	�Yv��A�K*

logging/current_costw�;��;-+       ��K	��v��A�K*

logging/current_cost]w�;�U��+       ��K	K�v��A�K*

logging/current_costAx�;���+       ��K	a�v��A�K*

logging/current_cost�v�;�0�L+       ��K	�w��A�K*

logging/current_costKx�;�~��+       ��K	�@w��A�K*

logging/current_costKw�;��@�+       ��K	Amw��A�K*

logging/current_costRw�;����+       ��K	�w��A�K*

logging/current_costjy�;��)�+       ��K	��w��A�K*

logging/current_costw�;UG�+       ��K	>�w��A�K*

logging/current_cost�w�;!�[4+       ��K	�x��A�K*

logging/current_costvw�;��+       ��K	�Lx��A�K*

logging/current_cost	w�;�:	+       ��K	�|x��A�K*

logging/current_costiw�;�QB�+       ��K	��x��A�K*

logging/current_cost)x�;Ď�6+       ��K	��x��A�K*

logging/current_cost�v�;w�n�+       ��K	vy��A�K*

logging/current_cost*x�;�ת+       ��K	�:y��A�K*

logging/current_costRw�;�G��+       ��K	[ky��A�K*

logging/current_costcw�;�JP�+       ��K	=�y��A�K*

logging/current_cost�y�;���U+       ��K	P�y��A�K*

logging/current_costNw�;���+       ��K	P�y��A�K*

logging/current_cost�x�; �+       ��K	� z��A�K*

logging/current_costhw�;��[+       ��K	wNz��A�L*

logging/current_costPw�;Ҫ>�+       ��K	7{z��A�L*

logging/current_costQw�;|n8+       ��K	m�z��A�L*

logging/current_cost�y�;�d+       ��K	�z��A�L*

logging/current_cost�w�;Ob!�+       ��K	S{��A�L*

logging/current_cost�w�;z�]+       ��K	�>{��A�L*

logging/current_costw�;C(+       ��K	"�{��A�L*

logging/current_cost�y�;y�H+       ��K	~�{��A�L*

logging/current_costxw�;�jk�+       ��K	)&|��A�L*

logging/current_cost|w�;�D�k+       ��K	be|��A�L*

logging/current_cost0w�;�h��+       ��K	,�|��A�L*

logging/current_cost-y�;�Y+       ��K	��|��A�L*

logging/current_cost�w�;��W+       ��K	1}��A�L*

logging/current_cost�w�;}7:#+       ��K	�U}��A�L*

logging/current_costw�;���#+       ��K	�}��A�L*

logging/current_costy�;9Dt~+       ��K	�}��A�L*

logging/current_cost�w�;�/Ս+       ��K	��}��A�L*

logging/current_cost}w�;�LT{+       ��K	�2~��A�L*

logging/current_cost w�;|���+       ��K	�n~��A�L*

logging/current_cost y�;�*J+       ��K	Ţ~��A�L*

logging/current_cost�w�;�ƹO+       ��K	��~��A�L*

logging/current_cost�w�;�H�+       ��K	7��A�L*

logging/current_cost�v�;xcmF+       ��K	�3��A�L*

logging/current_cost�x�;?���+       ��K	fh��A�L*

logging/current_cost�w�;�dBQ+       ��K	Ж��A�L*

logging/current_costow�;_���+       ��K	����A�L*

logging/current_costw�;��SG+       ��K	����A�M*

logging/current_cost�x�;���S+       ��K	y���A�M*

logging/current_cost|w�;�?k�+       ��K	�L���A�M*

logging/current_costcw�;r�1w+       ��K	�}���A�M*

logging/current_costw�;G(	+       ��K	o����A�M*

logging/current_cost�x�;�(+       ��K	�ۀ��A�M*

logging/current_costaw�;�73{+       ��K	����A�M*

logging/current_cost_w�;ٟk�+       ��K	�6���A�M*

logging/current_costrw�;I�+       ��K	�z���A�M*

logging/current_cost�w�;����+       ��K	 ����A�M*

logging/current_cost�v�;�{�+       ��K	,끲�A�M*

logging/current_costSw�;"��G+       ��K	@%���A�M*

logging/current_cost�v�;�P�+       ��K	'f���A�M*

logging/current_cost9w�;C�<�+       ��K	t����A�M*

logging/current_cost	w�;h�7 +       ��K	�ۂ��A�M*

logging/current_costjw�;uSڬ+       ��K	[���A�M*

logging/current_cost`w�;ׂ�'+       ��K	�D���A�M*

logging/current_cost-w�;)�+       ��K	,����A�M*

logging/current_cost�v�;���"+       ��K	˵���A�M*

logging/current_cost�v�;��+       ��K	惲�A�M*

logging/current_cost9w�;-�l+       ��K	#���A�M*

logging/current_cost�v�;�uxF+       ��K	�Q���A�M*

logging/current_cost�v�;��-+       ��K	6����A�M*

logging/current_cost�v�;��=�+       ��K	�����A�M*

logging/current_cost�v�;���+       ��K	�脲�A�M*

logging/current_cost�v�;7�
�+       ��K	����A�N*

logging/current_costw�;X��t+       ��K	F���A�N*

logging/current_costw�;��"�+       ��K	s���A�N*

logging/current_cost�x�;��]%